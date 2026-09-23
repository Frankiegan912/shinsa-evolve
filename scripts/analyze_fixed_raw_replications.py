#!/usr/bin/env python3
"""Summarize the six protocol-v4 fixed-raw LunarLander outer runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


RUN_DIRS = {
    ("claimed", 101): "pilot_claimed_seed101_retry",
    ("audited", 101): "pilot_audited_seed101",
    ("claimed", 202): "claimed_seed202",
    ("audited", 202): "audited_seed202",
    ("claimed", 303): "claimed_seed303",
    ("audited", 303): "audited_seed303",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_median(values: list[float]) -> dict:
    return {"values": values, "mean": statistics.mean(values),
            "median": statistics.median(values)}


def load_run(root: Path, mode: str, seed: int, dirname: str) -> dict:
    run_dir = root / dirname
    archive_path = run_dir / "archive.jsonl"
    config = json.loads((run_dir / "run_config.json").read_text())
    rows = [json.loads(line) for line in archive_path.read_text().splitlines() if line.strip()]
    done = [row for row in rows if row.get("status") == "done"]
    failed = [row for row in rows if row.get("status") != "done"]
    if len(done) != 15 or failed:
        raise RuntimeError(f"expected 15 done and 0 failed in {run_dir}")
    if (config.get("mode"), config.get("rng_seed")) != (mode, seed):
        raise RuntimeError(f"mode/seed mismatch in {run_dir}")
    if config.get("training_reward") != "fixed_raw":
        raise RuntimeError(f"not a fixed-raw run: {run_dir}")
    if config.get("mutator", {}).get("protocol_version") != 4:
        raise RuntimeError(f"not mutation protocol v4: {run_dir}")
    if any(len(row["audit"]["seeds"]) != 32 or
           len(row["audit"]["per_episode"]) != 32 for row in done):
        raise RuntimeError(f"incomplete audits in {run_dir}")
    if any(not math.isclose(row["claimed"], row["train"]["train_raw_of_best"],
                            rel_tol=0, abs_tol=1e-9) for row in done):
        raise RuntimeError(f"claim is not fixed raw training reward in {run_dir}")

    selected = max(done, key=lambda row: row[mode])
    oracle = max(done, key=lambda row: row["audited"])
    best_history = selected["train"]["history"][selected["train"]["best_generation"]]
    attempts = [attempt for row in done
                for attempt in (row.get("mutate") or {}).get("attempts", [])]
    return {
        "mode": mode,
        "rng_seed": seed,
        "run_dir": dirname,
        "archive_sha256": sha256(archive_path),
        "done": len(done),
        "failed": len(failed),
        "prompt_sha256": config["mutator"]["prompt_sha256"],
        "mutator": config["mutator"],
        "selected": {
            "id": selected["id"],
            "claimed": selected["claimed"],
            "audited": selected["audited"],
            "gap": selected["gap"],
            "best_generation": selected["train"]["best_generation"],
            "episodes_per_eval": best_history["episodes_per_eval"],
            "max_steps": best_history["max_steps"],
        },
        "oracle_by_audit": {"id": oracle["id"], "audited": oracle["audited"]},
        "selection_regret": oracle["audited"] - selected["audited"],
        "selection_error": selected["id"] != oracle["id"],
        "mutation": {
            "requests": len(attempts),
            "rejected_attempts": sum(not attempt.get("ok", False) for attempt in attempts),
            "input_tokens": sum(attempt.get("usage", {}).get("input_tokens", 0)
                                for attempt in attempts),
            "output_tokens": sum(attempt.get("usage", {}).get("output_tokens", 0)
                                 for attempt in attempts),
        },
    }


def summarize(root: Path) -> dict:
    runs = [load_run(root, mode, seed, dirname)
            for (mode, seed), dirname in RUN_DIRS.items()]
    prompt_hashes = {run["prompt_sha256"] for run in runs}
    identities = {json.dumps(run["mutator"], sort_keys=True) for run in runs}
    if len(prompt_hashes) != 1 or len(identities) != 1:
        raise RuntimeError("prompt or mutator identity differs across runs")

    groups = {}
    for mode in ("audited", "claimed"):
        group = [run for run in runs if run["mode"] == mode]
        groups[mode] = {
            "independent_outer_runs": len(group),
            "selected_audited": mean_median([run["selected"]["audited"] for run in group]),
            "selection_regret": mean_median([run["selection_regret"] for run in group]),
            "selection_errors": sum(run["selection_error"] for run in group),
        }
    by_mode_seed = {(run["mode"], run["rng_seed"]): run for run in runs}
    differences = [
        by_mode_seed[("audited", seed)]["selected"]["audited"]
        - by_mode_seed[("claimed", seed)]["selected"]["audited"]
        for seed in (101, 202, 303)
    ]
    return {
        "schema_version": 1,
        "scope": {
            "environment": "lunarlander",
            "training_reward": "fixed_raw",
            "candidates_per_run": 15,
            "modes": ["audited", "claimed"],
            "rng_seeds": [101, 202, 303],
            "independent_outer_runs_per_mode": 3,
            "inference_boundary": "descriptive ablation; n=3 per mode",
        },
        "protocol": {"mutation_protocol_version": 4,
                     "prompt_sha256": next(iter(prompt_hashes)),
                     "mutator": runs[0]["mutator"]},
        "runs": runs,
        "groups": groups,
        "matched_seed_descriptive_difference_audited_minus_claimed":
            mean_median(differences),
        "mutation_total": {
            "requests": sum(run["mutation"]["requests"] for run in runs),
            "rejected_attempts": sum(run["mutation"]["rejected_attempts"] for run in runs),
            "input_tokens": sum(run["mutation"]["input_tokens"] for run in runs),
            "output_tokens": sum(run["mutation"]["output_tokens"] for run in runs),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.runs_root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
