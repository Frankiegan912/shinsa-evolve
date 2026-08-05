#!/bin/bash
# End-to-end smoke: one seed candidate per env under a tiny budget (no LLM calls).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== breakout smoke ==="
python -m shinsa_evolve.core.orchestrate --env breakout --mode audited \
    --run-dir runs/smoke_breakout --smoke

echo "=== lunarlander smoke ==="
python -m shinsa_evolve.core.orchestrate --env lunarlander --mode audited \
    --run-dir runs/smoke_lunarlander --smoke

echo "=== smoke OK ==="
