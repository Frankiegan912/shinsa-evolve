#!/bin/bash
# Full experiment matrix (run overnight; each run resumes from its archive if interrupted).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m shinsa_evolve.core.orchestrate --env breakout    --mode audited --candidates 60 --run-dir runs/breakout_audited
python -m shinsa_evolve.core.orchestrate --env breakout    --mode claimed --candidates 20 --run-dir runs/breakout_claimed
python -m shinsa_evolve.core.orchestrate --env lunarlander --mode audited --candidates 40 --run-dir runs/lunar_audited
python -m shinsa_evolve.core.orchestrate --env lunarlander --mode claimed --candidates 15 --run-dir runs/lunar_claimed

# Seed-recipe variance baseline (3 independent retrains per env).
python -m shinsa_evolve.core.orchestrate --env breakout    --seed-variance 3 --run-dir runs/breakout_seed_variance
python -m shinsa_evolve.core.orchestrate --env lunarlander --seed-variance 3 --run-dir runs/lunar_seed_variance

python -m shinsa_evolve.analyze.figures --runs runs/breakout_audited runs/breakout_claimed \
    runs/lunar_audited runs/lunar_claimed --out analysis_out
