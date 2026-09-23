# Protocol-v3 LunarLander claim audit

This report defines the evidence boundary for the primary preprint experiment. The unit of
replication is one complete outer-loop archive, not a candidate, training episode, or audit seed.

## Evidence and reconstruction

Six protocol-v3 archives contain three audited-selection and three claimed-selection runs, using
outer RNG seeds 101, 202, and 303. Each archive has 15 completed candidates and no failed records.
Every candidate stores its recipe, trained parameters, training seed/history, 32 fresh audit
seeds, per-episode raw returns, and claimed/audited scores.

Recompute the machine-readable summary without executing generated code:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/analyze_replications.py \
  --runs-root runs/protocol_v3_codex/lunar_replications \
  --output evidence/protocol_v3_lunar_summary.json
```

The script checks run completeness, archive schema, mode/seed configuration, audit-seed count,
mutation protocol version, prompt hash, and provider identity before producing statistics.

## Supported descriptive claims

| Quantity | Audited selection | Claimed selection |
|---|---:|---:|
| Independent outer runs | 3 | 3 |
| Final-pick audits | 33.00, -48.73, 27.62 | -1102.33, -422.71, -761.39 |
| Mean final-pick audit | 3.96 | -762.14 |
| Within-archive selection errors | 0 by definition | 3 of 3 |
| Mean actual selection regret | 0 by definition | 710.36 |
| Mean minimum regret over top-score ties | 0 by definition | 204.09 |

The seed-aligned audited-minus-claimed final-pick differences are 1135.33, 373.97, and 789.01
raw-reward points. They are descriptive: model outputs and OS-entropy audit seeds are not paired.

Claimed-selection candidates reached a shaped training score of exactly 1,000,000,000 in every
run. The top claim was tied by 5, 2, and 3 candidates. This supports the mechanism claim that
candidate-controlled reward scale can saturate and lose ranking information. It does not support
the claim that every high-claim candidate is poor: one seed-303 candidate claimed approximately
999.9 million and had the best audit in its archive (41.56).

## Provenance

- mutation protocol: 3;
- provider: Codex CLI 0.146.0, `gpt-5.6-sol`, low reasoning;
- prompt SHA-256: `be9c5020a26d3bae12e5ec8ec5f2e3b2febfa34d15ab2d9583ede47289a00e8a`;
- mutation requests: 84, all accepted on the first attempt;
- recorded usage: 1,490,447 input tokens and 46,924 output tokens.

## Unsupported or not-yet-verifiable claims

- statistical significance, expected effect size, or population frequency with only three runs
  per condition;
- cross-environment replication: Breakout remains a historical single-run pilot;
- generality across mutation models, model versions, or program-evolution domains;
- stable recipe improvement, because selected recipes were not independently retrained;
- a unique causal decomposition among reward scaling, schedule changes, seed reuse, and training
  noise;
- immutable model weights behind the recorded service model identifier.

## Public-release boundary

The protocol-v3 bundle contains only this project's generated recipes, parameters, configurations,
and recorded results. Machine-local seed-recipe paths are sanitized. Restricted external material,
credentials, caches, smoke runs, and editor state are excluded. The bundle is scanned before
release and carries per-file hashes plus a deterministic archive checksum.
