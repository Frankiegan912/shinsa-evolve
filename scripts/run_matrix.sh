#!/bin/bash
# Full experiment matrix (run overnight; each run resumes from its archive if interrupted).
set -euo pipefail
cd "$(dirname "$0")/.."

export SHINSA_MUTATE_PROVIDER=codex
export SHINSA_MUTATE_MODEL=gpt-5.6-sol
export SHINSA_MUTATE_REASONING=low
run_root="${SHINSA_RUN_ROOT:-runs/protocol_v3_codex}"
mkdir -p "$run_root"

python -m shinsa_evolve.core.orchestrate --env breakout    --mode audited --candidates 60 --run-dir "$run_root/breakout_audited"
python -m shinsa_evolve.core.orchestrate --env breakout    --mode claimed --candidates 20 --run-dir "$run_root/breakout_claimed"
python -m shinsa_evolve.core.orchestrate --env lunarlander --mode audited --candidates 40 --run-dir "$run_root/lunar_audited"
python -m shinsa_evolve.core.orchestrate --env lunarlander --mode claimed --candidates 15 --run-dir "$run_root/lunar_claimed"

# Seed-recipe variance baseline (3 independent retrains per env).
python -m shinsa_evolve.core.orchestrate --env breakout    --seed-variance 3 --run-dir "$run_root/breakout_seed_variance"
python -m shinsa_evolve.core.orchestrate --env lunarlander --seed-variance 3 --run-dir "$run_root/lunar_seed_variance"

python -m shinsa_evolve.analyze.figures --runs "$run_root/breakout_audited" "$run_root/breakout_claimed" \
    "$run_root/lunar_audited" "$run_root/lunar_claimed" --out "$run_root/analysis_out"
