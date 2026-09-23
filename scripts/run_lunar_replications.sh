#!/bin/bash
# Protocol-v2 causal replication: three independent outer loops per selection mode.
# This script makes external Codex CLI calls against the logged-in ChatGPT account.
set -euo pipefail
cd "$(dirname "$0")/.."

export SHINSA_MUTATE_PROVIDER=codex
export SHINSA_MUTATE_MODEL=gpt-5.6-sol
export SHINSA_MUTATE_REASONING=low
run_root="${SHINSA_RUN_ROOT:-runs/protocol_v3_codex/lunar_replications}"
python_bin="${SHINSA_PYTHON:-.venv/bin/python}"
mkdir -p "$run_root"

for rng_seed in 101 202 303; do
  "$python_bin" -m shinsa_evolve.core.orchestrate \
    --env lunarlander --mode audited --candidates 15 --rng-seed "$rng_seed" \
    --run-dir "$run_root/audited_seed${rng_seed}"
  "$python_bin" -m shinsa_evolve.core.orchestrate \
    --env lunarlander --mode claimed --candidates 15 --rng-seed "$rng_seed" \
    --run-dir "$run_root/claimed_seed${rng_seed}"
done
