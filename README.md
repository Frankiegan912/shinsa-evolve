# shinsa-evolve

Can you trust the scores an LLM-driven program-evolution loop reports about its own
candidates? This repository studies that question for *training recipes*: small Python
modules, written by an LLM mutation operator, that control reward shaping, evaluation
schedule, and optimizer settings for a fixed inner training loop (small MLP + separable
CMA-ES).

Every candidate gets two scores:

- **claimed** — the best training fitness under the recipe's own measurement (its shaped
  reward, its episode counts, a fixed training seed pool);
- **audited** — an independent re-evaluation of the trained policy on freshly drawn OS-entropy
  seeds, scored by raw environment reward only, with no execution path for recipe code.

The difference is the **honesty gap**. The experiment's single manipulated variable is which
score drives selection (`--mode audited` vs `--mode claimed`); audits always run for
recording. Environments: MinAtar Breakout (gymnax, jit batch rollouts) and LunarLander-v3
(gymnasium, spawn process pool) — two backends on purpose, so the harness claim does not
depend on one execution model.

This is a clean-room, from-scratch implementation; it contains no code, prompts, data, or
text from any prior private implementation.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install swig && pip install "gymnasium[box2d]"   # box2d needs swig at build time
```

## Smoke test (no LLM calls)

```bash
bash scripts/smoke.sh
```

One-mutation LLM smoke (uses the `claude` CLI once):

```bash
python -m shinsa_evolve.core.orchestrate --env breakout --mode audited \
    --run-dir runs/smoke_llm --smoke-llm
```

## Full matrix

```bash
bash scripts/run_matrix.sh
```

Runs are resumable: rerunning the same command continues from `archive.jsonl`.

LLM mutation uses the `claude` CLI (`claude -p`); override with `SHINSA_CLAUDE_BIN`
and `SHINSA_MUTATE_MODEL`.

Note: recipe code is LLM-generated Python executed in-process. Static checks reject
IO/imports as hygiene, but this is not a security sandbox — run on infrastructure you
trust with nothing sensitive in the environment.

License: TBD before public release.
