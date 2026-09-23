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
import importlib.metadata
import json
import platform
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
RUN_CONFIG_SCHEMA = 2
RNG_STATE_SCHEMA = 1
RUNTIME_DISTRIBUTIONS = (
    "numpy", "cmaes", "gymnasium", "gymnax", "jax", "jaxlib", "matplotlib", "Box2D",
)


class FixedRawTrainingAdapter:
    """Use harness-owned raw reward for training, without calling recipe shaping."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def eval_shaped(self, members, seeds, max_steps, recipe):
        class RawRecipe:
            @staticmethod
            def shaped_reward(obs, action, reward, terminated, truncated, step):
                return reward

        _, raw = self.base.eval_shaped(members, seeds, max_steps, RawRecipe)
        return raw, raw

    def close(self):
        self.base.close()


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def runtime_metadata() -> dict:
    versions = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": versions,
    }


def ensure_run_config(run_dir: Path, desired: dict) -> None:
    """Create an immutable run config, or reject an incompatible resume."""
    path = run_dir / "run_config.json"
    if not path.exists():
        _atomic_write_json(path, desired)
        return
    try:
        existing = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid existing run config: {path}: {e}") from e
    if existing != desired:
        keys = sorted(key for key in set(existing) | set(desired)
                      if existing.get(key) != desired.get(key))
        raise RuntimeError(
            f"refusing incompatible resume in {run_dir}; changed config keys: {keys}"
        )


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def save_outer_rng(run_dir: Path, rng_seed: int, rng: random.Random) -> None:
    _atomic_write_json(run_dir / "outer_rng_state.json", {
        "schema_version": RNG_STATE_SCHEMA,
        "rng_seed": rng_seed,
        "state": rng.getstate(),
    })


def load_outer_rng(run_dir: Path, rng_seed: int, archive: Archive) -> random.Random:
    """Restore the exact outer RNG state, refusing ambiguous legacy resumes."""
    path = run_dir / "outer_rng_state.json"
    rng = random.Random(rng_seed)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != RNG_STATE_SCHEMA:
            raise RuntimeError(f"unsupported outer RNG state schema in {path}")
        if payload.get("rng_seed") != rng_seed:
            raise RuntimeError(f"outer RNG seed mismatch in {path}")
        rng.setstate(_nested_tuple(payload["state"]))
        return rng
    if any(record.get("parent") is not None for record in archive.load()):
        raise RuntimeError(
            f"cannot resume {run_dir}: outer-loop records exist but RNG state is missing"
        )
    save_outer_rng(run_dir, rng_seed, rng)
    return rng


def record_candidate_failure(archive, cid, parent_id, env_name, mode, stage,
                             error, recipe_path, mutate_log):
    """Append a compact failure record without embedding a machine-local traceback."""
    archive.append({
        "schema_version": 2,
        "id": cid,
        "parent": parent_id,
        "env": env_name,
        "mode": mode,
        "status": "candidate_failed",
        "stage": stage,
        "recipe": str(Path(recipe_path).relative_to(archive.run_dir)),
        "mutate": mutate_log,
        "error_type": type(error).__name__,
        "error": str(error)[:500],
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    })


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
        "schema_version": 2,
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
    (archive.run_dir / "params" ).mkdir(exist_ok=True)
    import numpy as np
    np.save(archive.run_dir / "params" / f"cand_{cid:03d}.npy", tr.best_params)
    archive.append(record)
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
    p.add_argument("--training-reward", default="candidate_shaped",
                   choices=["candidate_shaped", "fixed_raw"],
                   help="training fitness source; fixed_raw is the reward-scale control")
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
    seed_src = Path(args.seed_recipe or DEFAULT_SEEDS[args.env])
    archive = Archive(run_dir)
    uses_llm = args.seed_variance == 0 and candidates > 1
    if uses_llm:
        mutator = (llm_mutate.mutator_identity(training_reward="fixed_raw")
                   if args.training_reward == "fixed_raw"
                   else llm_mutate.mutator_identity())
    else:
        mutator = None
    desired_config = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "env": args.env, "mode": args.mode, "candidates": candidates,
        "budget": vars(budget), "audit_episodes": audit_episodes,
        "seed_recipe": str(seed_src), "rng_seed": args.rng_seed,
        "pool": args.pool, "seed_variance": args.seed_variance,
        "smoke": args.smoke, "smoke_llm": args.smoke_llm,
        "runtime": runtime_metadata(),
        "mutator": mutator,
    }
    if args.training_reward != "candidate_shaped":
        desired_config["training_reward"] = args.training_reward
    ensure_run_config(run_dir, desired_config)
    (run_dir / "recipes").mkdir(exist_ok=True)
    rng = load_outer_rng(run_dir, args.rng_seed, archive)
    adapter = get_adapter(args.env, pool_size=args.pool)
    if args.training_reward == "fixed_raw":
        adapter = FixedRawTrainingAdapter(adapter)

    try:
        if archive.count_done() == 0:
            seed_cid = archive.next_id()
            path = run_dir / "recipes" / f"cand_{seed_cid:03d}.py"
            shutil.copyfile(seed_src, path)
            log(f"seed candidate from {seed_src.name}")
            try:
                process_candidate(seed_cid, path, None, None, adapter, budget, audit_episodes,
                                  archive, args.mode, args.env, train_seed=args.rng_seed * 1000)
            except Exception as e:
                record_candidate_failure(
                    archive, seed_cid, None, args.env, args.mode, "seed_candidate",
                    e, path, None,
                )
                log(f"seed candidate failed: {type(e).__name__}: {e}")
                return 1

        if args.seed_variance > 0:
            # Idempotent: target is 1 seed + N retrains, so retry wrappers can rerun safely.
            seed_record = min(archive.done(), key=lambda r: r["id"])
            seed_path = run_dir / seed_record["recipe"]
            while archive.count_done() < 1 + args.seed_variance:
                idx = archive.count_done()
                cid = archive.next_id()
                mutate_log = {"variance_retrain": True}
                try:
                    process_candidate(cid, seed_path, seed_record["id"], mutate_log, adapter,
                                      budget, audit_episodes, archive, args.mode, args.env,
                                      train_seed=args.rng_seed * 1000 + idx * 97)
                except Exception as e:
                    record_candidate_failure(
                        archive, cid, seed_record["id"], args.env, args.mode,
                        "seed_variance_retrain", e, seed_path, mutate_log,
                    )
                    log(f"seed retrain {cid:03d} failed: {type(e).__name__}: {e}")
                    return 1
            log(f"seed-variance complete ({args.seed_variance} retrains)")
            return 0

        consecutive_failures = 0
        while archive.count_done() < candidates:
            cid = archive.next_id()
            parent = select_parent(archive, args.mode, rng)
            save_outer_rng(run_dir, args.rng_seed, rng)
            parent_src = (run_dir / parent["recipe"]).read_text()
            feedback = parent["claimed"] if args.mode == "claimed" else parent["audited"]
            path = run_dir / "recipes" / f"cand_{cid:03d}.py"
            try:
                mutate_kwargs = ({"training_reward": "fixed_raw"}
                                 if args.training_reward == "fixed_raw" else {})
                _, attempts = llm_mutate.mutate(
                    parent_src, feedback, cfg["description"], args.mode,
                    adapter.obs_dim, path, **mutate_kwargs,
                )
            except llm_mutate.ProviderUnavailable as e:
                archive.append({"id": cid, "parent": parent["id"], "env": args.env,
                                "mode": args.mode, "status": "mutate_failed",
                                "error": str(e)[:500],
                                "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                log(f"cand {cid:03d} mutation provider unavailable; aborting run")
                return 1
            except llm_mutate.MutationExhausted as e:
                consecutive_failures += 1
                archive.append({"schema_version": 2, "id": cid, "parent": parent["id"],
                                "env": args.env, "mode": args.mode,
                                "status": "mutate_failed", "error": str(e),
                                "mutate": {"attempts": e.attempts,
                                           "parent_feedback": feedback},
                                "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                log(f"cand {cid:03d} mutation failed ({consecutive_failures} in a row)")
                if consecutive_failures >= 3:
                    log("aborting: 3 consecutive mutation failures (check mutation protocol)")
                    return 1
                continue
            except llm_mutate.MutateError as e:
                consecutive_failures += 1
                archive.append({"id": cid, "parent": parent["id"], "env": args.env,
                                "mode": args.mode, "status": "mutate_failed",
                                "error": str(e)[:500],
                                "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                log(f"cand {cid:03d} mutation failed ({consecutive_failures} in a row)")
                if consecutive_failures >= 3:
                    log("aborting: 3 consecutive mutation failures (check mutation provider)")
                    return 1
                continue
            consecutive_failures = 0
            mlog = {"attempts": attempts, "parent_feedback": feedback}
            try:
                process_candidate(cid, path, parent["id"], mlog, adapter, budget,
                                  audit_episodes, archive, args.mode, args.env,
                                  train_seed=args.rng_seed * 1000 + cid)
            except Exception as e:
                # Preserve the failed candidate ID and recipe so a resumed run does not
                # silently overwrite provenance. KeyboardInterrupt/SystemExit still escape.
                record_candidate_failure(
                    archive, cid, parent["id"], args.env, args.mode,
                    "evolved_candidate", e, path, mlog,
                )
                log(f"cand {cid:03d} failed during training or audit: {type(e).__name__}: {e}")

        best = archive.top(args.mode, 1)[0]
        log(f"run complete: {archive.count_done()} candidates, "
            f"best-by-{args.mode} = cand {best['id']:03d} "
            f"(claimed={best['claimed']:.2f}, audited={best['audited']:.2f})")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    sys.exit(main())
