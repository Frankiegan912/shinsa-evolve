#!/usr/bin/env python3
"""Estimate protocol-v2 call count, API-equivalent cost, and wall time offline.

The estimate uses prompt sizes and mutation replies from the four recovered selection
archives. It makes no network or model calls. A ChatGPT-authenticated Codex CLI run consumes
subscription allowance rather than the API-equivalent dollar prices supplied here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from shinsa_evolve.core.llm_mutate import build_prompt
from shinsa_evolve.envs import ENV_CONFIGS


RUN_NAMES = (
    "breakout_audited",
    "breakout_claimed",
    "lunar_audited",
    "lunar_claimed",
)
REPLICATION_RUNS = 6
CANDIDATES_PER_RUN = 15
MAX_ATTEMPTS = 4


def recovered_attempts(repo: Path) -> list[dict]:
    samples = []
    for name in RUN_NAMES:
        run = repo / "runs" / name
        rows = [json.loads(line) for line in (run / "archive.jsonl").read_text().splitlines()]
        done = {row["id"]: row for row in rows if row.get("status") == "done"}
        for row in rows:
            mutation = row.get("mutate")
            parent_id = row.get("parent")
            if not mutation or parent_id is None or "attempts" not in mutation:
                continue
            parent = done[parent_id]
            parent_source = (run / parent["recipe"]).read_text()
            feedback = row["claimed"] if row["mode"] == "claimed" else row["audited"]
            prompt_chars = len(build_prompt(
                parent_source,
                feedback,
                ENV_CONFIGS[row["env"]]["description"],
                row["mode"],
            ))
            for attempt in mutation["attempts"]:
                samples.append({
                    "prompt_chars": prompt_chars,
                    "reply_chars": attempt.get("reply_chars", 0),
                    "seconds": attempt.get("seconds", 0.0),
                    "mutation_id": (name, row["id"]),
                })
    return samples


def dollars(prompt_chars: float, reply_chars: float, calls: int, chars_per_token: float,
            input_per_million: float, output_per_million: float) -> float:
    input_tokens = prompt_chars / chars_per_token
    output_tokens = reply_chars / chars_per_token
    return calls * (
        input_tokens * input_per_million / 1_000_000
        + output_tokens * output_per_million / 1_000_000
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-per-million", type=float, default=2.0)
    parser.add_argument("--output-per-million", type=float, default=10.0)
    args = parser.parse_args()

    samples = recovered_attempts(args.repo_root.resolve())
    mutation_count = len({sample["mutation_id"] for sample in samples})
    required_successes = REPLICATION_RUNS * (CANDIDATES_PER_RUN - 1)
    expected_calls = math.ceil(required_successes * len(samples) / mutation_count)
    maximum_calls = required_successes * MAX_ATTEMPTS
    mean_prompt = statistics.mean(sample["prompt_chars"] for sample in samples)
    mean_reply = statistics.mean(sample["reply_chars"] for sample in samples)
    mean_seconds = statistics.mean(sample["seconds"] for sample in samples)
    max_prompt = max(sample["prompt_chars"] for sample in samples)
    max_reply = max(sample["reply_chars"] for sample in samples)

    central_cost = dollars(
        mean_prompt, mean_reply, expected_calls, 4.0,
        args.input_per_million, args.output_per_million,
    )
    conservative_cost = dollars(
        max_prompt, max_reply, maximum_calls, 3.0,
        args.input_per_million, args.output_per_million,
    )
    print(f"historical mutations/attempts: {mutation_count}/{len(samples)}")
    print(f"required successful mutations: {required_successes}")
    print(f"expected CLI requests at historical retry rate: {expected_calls}")
    print(f"maximum CLI requests at {MAX_ATTEMPTS} attempts each: {maximum_calls}")
    print(f"central API-equivalent cost: ${central_cost:.2f}")
    print(f"conservative all-retry cost bound: ${conservative_cost:.2f}")
    print(f"expected sequential mutation wall time: {mean_seconds * expected_calls / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
