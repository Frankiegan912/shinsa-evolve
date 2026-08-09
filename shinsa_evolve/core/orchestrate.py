"""Outer evolution loop: select parent -> mutate -> train -> audit -> archive.

Selection mode is the experiment's single manipulated variable:
  audited  parents are chosen by audited score; the LLM sees audited feedback
  claimed  parents are chosen by the recipe's self-reported training score; the
           LLM sees claimed feedback. Audits still run on every candidate, but
           only as recording, never as selection input.

The archive is the resumable state: rerunning the same command continues a run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
import sys
import time
from pathlib import Path

from ..envs import ENV_CONFIGS, get_adapter
from . import llm_mutate
from .archive import Archive
from .audit import audit
from .inner_train import TrainBudget, train
from .interface import RecipeError, load_recipe, validate_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = {
    "breakout": REPO_ROOT / "recipes" / "seed_breakout.py",
    "lunarlander": REPO_ROOT / "recipes" / "seed_lunarlander.py",
}
EXPLOIT_TOP_K = 5
EXPLOIT_PROB = 0.8


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def select_parent(archive: Archive, mode: str, rng: random.Random) -> dict:
    done = archive.done()
    if not done:
        raise RuntimeError("no completed candidates to select a parent from")
    if rng.random() < EXPLOIT_PROB:
        return rng.choice(archive.top(mode, EXPLOIT_TOP_K))
    return rng.choice(done)


def process_candidate(cid, recipe_path, parent_id, mutate_log, adapter, budget,
                      audit_episodes, archive, mode, env_name, train_seed):
    recipe = load_recipe(recipe_path)
    problems = validate_recipe(recipe, adapter.obs_dim)
    if problems:
        raise RecipeError(problems)
    t0 = time.monotonic()
    tr = train(recipe, adapter, budget, train_seed=train_seed)
    au = audit(tr.best_params, adapter, episodes=audit_episodes)
    record = {
        "id": cid,
        "parent": parent_id,
        "env": env_name,
        "mode": mode,
        "status": "done",
        "claimed": tr.claimed_score,
        "audited": au.score,
        "gap": tr.claimed_score - au.score,
        "recipe": str(Path(recipe_path).relative_to(archive.run_dir)),
        "train": tr.stats(),
        "audit": au.stats(),
        "mutate": mutate_log,
        "total_seconds": round(time.monotonic() - t0, 1),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    archive.append(record)
    (archive.run_dir / "params" ).mkdir(exist_ok=True)
    import numpy as np
    np.save(archive.run_dir / "params" / f"cand_{cid:03d}.npy", tr.best_params)
    log(
        f"cand {cid:03d} (parent {parent_id}) claimed={tr.claimed_score:.2f} "
        f"audited={au.score:.2f} gap={tr.claimed_score - au.score:+.2f} "
        f"gens={tr.generations} eps={tr.episodes_used} wall={tr.wall_seconds:.0f}s"
    )
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description="shinsa-evolve orchestrator")
    p.add_argument("--env", required=True, choices=list(ENV_CONFIGS))
    p.add_argument("--mode", default="audited", choices=["audited", "claimed"])
    p.add_argument("--candidates", type=int, default=60)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed-recipe", default=None)
    p.add_argument("--rng-seed", type=int, default=1)
    p.add_argument("--pool", type=int, default=None, help="worker processes (lunarlander)")
    p.add_argument("--max-wall", type=float, default=None, help="per-candidate wall budget (s)")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--audit-episodes", type=int, default=None)
    p.add_argument("--seed-variance", type=int, default=0,
                   help="instead of evolving: retrain the seed recipe N extra times")
    p.add_argument("--smoke", action="store_true", help="tiny budget, seed candidate only")
    p.add_argument("--smoke-llm", action="store_true",
                   help="tiny budget, seed + one LLM mutation")
    args = p.parse_args(argv)

    cfg = ENV_CONFIGS[args.env]
    budget = TrainBudget(
        max_episodes=args.max_episodes or cfg["max_episodes"],
        max_wall_seconds=args.max_wall or cfg["max_wall_seconds"],
    )
    audit_episodes = args.audit_episodes or cfg["audit_episodes"]
    candidates = args.candidates
    if args.smoke or args.smoke_llm:
        budget = TrainBudget(max_episodes=min(budget.max_episodes, 300),
                             max_wall_seconds=min(budget.max_wall_seconds, 90.0))
        audit_episodes = min(audit_episodes, 8)
        candidates = 2 if args.smoke_llm else 1

    run_dir = Path(args.run_dir)
    archive = Archive(run_dir)
    (run_dir / "recipes").mkdir(exist_ok=True)
    rng = random.Random(args.rng_seed)
    adapter = get_adapter(args.env, pool_size=args.pool)
    seed_src = Path(args.seed_recipe or DEFAULT_SEEDS[args.env])

    (run_dir / "run_config.json").write_text(json.dumps({
        "env": args.env, "mode": args.mode, "candidates": candidates,
        "budget": vars(budget), "audit_episodes": audit_episodes,
        "seed_recipe": str(seed_src), "rng_seed": args.rng_seed,
    }, indent=2))

    try:
        if archive.count_done() == 0:
            path = run_dir / "recipes" / "cand_000.py"
            shutil.copyfile(seed_src, path)
            log(f"seed candidate from {seed_src.name}")
            process_candidate(0, path, None, None, adapter, budget, audit_episodes,
                              archive, args.mode, args.env, train_seed=args.rng_seed * 1000)

        if args.seed_variance > 0:
            # Idempotent: target is 1 seed + N retrains, so retry wrappers can rerun safely.
            while archive.count_done() < 1 + args.seed_variance:
                idx = archive.count_done()
                cid = archive.next_id()
                path = run_dir / "recipes" / "cand_000.py"
                process_candidate(cid, path, 0, {"variance_retrain": True}, adapter,
                                  budget, audit_episodes, archive, args.mode, args.env,
                                  train_seed=args.rng_seed * 1000 + idx * 97)
            log(f"seed-variance complete ({args.seed_variance} retrains)")
            return 0

        consecutive_failures = 0
        while archive.count_done() < candidates:
            cid = archive.next_id()
            parent = select_parent(archive, args.mode, rng)
            parent_src = (run_dir / parent["recipe"]).read_text()
            feedback = parent["claimed"] if args.mode == "claimed" else parent["audited"]
            path = run_dir / "recipes" / f"cand_{cid:03d}.py"
            try:
                _, attempts = llm_mutate.mutate(
                    parent_src, feedback, cfg["description"], args.mode,
                    adapter.obs_dim, path,
                )
            except llm_mutate.MutateError as e:
                consecutive_failures += 1
                archive.append({"id": cid, "parent": parent["id"], "env": args.env,
                                "mode": args.mode, "status": "mutate_failed",
                                "error": str(e)[:500],
                                "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                log(f"cand {cid:03d} mutation failed ({consecutive_failures} in a row)")
                if consecutive_failures >= 3:
                    log("aborting: 3 consecutive mutation failures (check claude CLI)")
                    return 1
                continue
            consecutive_failures = 0
            mlog = {"attempts": attempts, "parent_feedback": feedback}
            try:
                process_candidate(cid, path, parent["id"], mlog, adapter, budget,
                                  audit_episodes, archive, args.mode, args.env,
                                  train_seed=args.rng_seed * 1000 + cid)
            except RecipeError as e:
                # Validated at mutation time; a failure here is unexpected but non-fatal.
                archive.append({"id": cid, "parent": parent["id"], "env": args.env,
                                "mode": args.mode, "status": "train_failed",
                                "error": str(e)[:500],
                                "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                log(f"cand {cid:03d} failed: {e}")

        best = archive.top(args.mode, 1)[0]
        log(f"run complete: {archive.count_done()} candidates, "
            f"best-by-{args.mode} = cand {best['id']:03d} "
            f"(claimed={best['claimed']:.2f}, audited={best['audited']:.2f})")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    sys.exit(main())
