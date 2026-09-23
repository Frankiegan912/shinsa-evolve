#!/bin/bash
# End-to-end smoke: one seed candidate per env under a tiny budget (no LLM calls).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${SHINSA_SMOKE_ROOT:-}" ]]; then
    smoke_root="$SHINSA_SMOKE_ROOT"
    mkdir -p "$smoke_root"
else
    smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/shinsa-evolve-smoke.XXXXXX")"
fi
export MPLCONFIGDIR="$smoke_root/matplotlib-cache"
export XDG_CACHE_HOME="$smoke_root/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
echo "smoke output: $smoke_root"

echo "=== breakout smoke ==="
python -m shinsa_evolve.core.orchestrate --env breakout --mode audited \
    --run-dir "$smoke_root/breakout" --smoke

echo "=== lunarlander smoke ==="
python -m shinsa_evolve.core.orchestrate --env lunarlander --mode audited \
    --run-dir "$smoke_root/lunarlander" --smoke --pool 2

echo "=== smoke OK ==="
