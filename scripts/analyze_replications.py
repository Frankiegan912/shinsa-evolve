#!/usr/bin/env python3
"""Summarize independent protocol-v3 outer-loop replications.

The unit of replication is one complete archive, never a candidate or audit episode.
This script executes no generated recipe code and makes no network/model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


EXPECTED_SEEDS = (101, 202, 303)
EXPECTED_MODES = ("audited", "claimed")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_median(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def load_run(run_dir: Path, mode: str, seed: int) -> dict:
    archive_path = run_dir / "archive.jsonl"
    config_path = run_dir / "run_config.json"
    if not archive_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"missing archive/config in {run_dir}")
    rows = [json.loads(line) for line in archive_path.read_text().splitlines() if line.strip()]
    done = [row for row in rows if row.get("status") == "done"]
    failed = [row for row in rows if row.get("status") != "done"]
    config = json.loads(config_path.read_text())
    if len(done) != 15 or failed:
        raise RuntimeError(f"expected 15 done and 0 failed in {run_dir}")
    if config.get("schema_version") != 2 or config.get("candidates") != 15:
        raise RuntimeError(f"unexpected run config in {run_dir}")
    if config.get("mode") != mode or config.get("rng_seed") != seed:
        raise RuntimeError(f"mode/seed mismatch in {run_dir}")
    if config.get("mutator", {}).get("protocol_version") != 3:
        raise RuntimeError(f"not mutation protocol v3: {run_dir}")
    if any(row.get("schema_version") != 2 for row in done):
        raise RuntimeError(f"non-v2 candidate record in {run_dir}")
    if any(len(row.get("audit", {}).get("seeds", [])) != 32 for row in done):
        raise RuntimeError(f"incomplete audit seeds in {run_dir}")

    selection_key = "audited" if mode == "audited" else "claimed"
    selected = sorted(done, key=lambda row: row[selection_key], reverse=True)[0]
    oracle = sorted(done, key=lambda row: row["audited"], reverse=True)[0]
    top_tied = [row for row in done if row[selection_key] == selected[selection_key]]
    best_audited_among_ties = max(top_tied, key=lambda row: row["audited"])
    attempts = [
        attempt
        for row in done
        for attempt in (row.get("mutate") or {}).get("attempts", [])
    ]
    return {
        "mode": mode,
        "rng_seed": seed,
        "done": len(done),
        "failed": len(failed),
        "archive_sha256": sha256(archive_path),
        "prompt_sha256": config["mutator"]["prompt_sha256"],
        "mutator": config["mutator"],
        "seed_candidate_audited": done[0]["audited"],
        "selected": {
            "id": selected["id"],
            "claimed": selected["claimed"],
            "audited": selected["audited"],
            "gap": selected["gap"],
        },
        "oracle_by_audit": {
            "id": oracle["id"],
            "audited": oracle["audited"],
        },
        "selection_regret": oracle["audited"] - selected["audited"],
        "selection_error": selected["id"] != oracle["id"],
        "top_score_ties": len(top_tied),
        "tie_robust": {
            "best_audited_id": best_audited_among_ties["id"],
            "best_audited": best_audited_among_ties["audited"],
            "minimum_selection_regret": (
                oracle["audited"] - best_audited_among_ties["audited"]
            ),
        },
        "mutation": {
            "requests": len(attempts),
            "rejected_attempts": sum(not attempt.get("ok", False) for attempt in attempts),
            "input_tokens": sum(
                attempt.get("usage", {}).get("input_tokens", 0) for attempt in attempts
            ),
            "output_tokens": sum(
                attempt.get("usage", {}).get("output_tokens", 0) for attempt in attempts
            ),
        },
    }


def summarize(root: Path) -> dict:
    runs = []
    for mode in EXPECTED_MODES:
        for seed in EXPECTED_SEEDS:
            runs.append(load_run(root / f"{mode}_seed{seed}", mode, seed))
    prompt_hashes = {run["prompt_sha256"] for run in runs}
    identities = {
        json.dumps(run["mutator"], sort_keys=True, separators=(",", ":")) for run in runs
    }
    if len(prompt_hashes) != 1 or len(identities) != 1:
        raise RuntimeError("mutation prompt or provider identity differs across runs")

    groups = {}
    for mode in EXPECTED_MODES:
        group = [run for run in runs if run["mode"] == mode]
        groups[mode] = {
            "independent_outer_runs": len(group),
            "selected_audited": mean_median([run["selected"]["audited"] for run in group]),
            "oracle_audited": mean_median(
                [run["oracle_by_audit"]["audited"] for run in group]
            ),
            "selection_regret": mean_median([run["selection_regret"] for run in group]),
            "minimum_tie_robust_selection_regret": mean_median(
                [run["tie_robust"]["minimum_selection_regret"] for run in group]
            ),
            "selected_honesty_gap": mean_median([run["selected"]["gap"] for run in group]),
            "selection_errors": sum(run["selection_error"] for run in group),
        }
    by_mode_seed = {(run["mode"], run["rng_seed"]): run for run in runs}
    differences = [
        by_mode_seed[("audited", seed)]["selected"]["audited"]
        - by_mode_seed[("claimed", seed)]["selected"]["audited"]
        for seed in EXPECTED_SEEDS
    ]
    return {
        "schema_version": 1,
        "scope": {
            "environment": "lunarlander",
            "candidates_per_run": 15,
            "modes": list(EXPECTED_MODES),
            "rng_seeds": list(EXPECTED_SEEDS),
            "independent_outer_runs_per_mode": 3,
            "inference_boundary": "descriptive; n=3 per mode",
        },
        "protocol": {
            "mutation_protocol_version": 3,
            "prompt_sha256": next(iter(prompt_hashes)),
            "mutator": runs[0]["mutator"],
        },
        "runs": runs,
        "groups": groups,
        "matched_seed_descriptive_difference_audited_minus_claimed": mean_median(differences),
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
    result = summarize(args.runs_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"replication summary written to {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
