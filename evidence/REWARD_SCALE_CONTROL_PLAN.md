# Fixed-raw reward-source ablation: preregistered plan and completed outcome

The prospective design below was recorded before the full replication matrix was completed. Later
sections preserve the execution history and report the completed outcome without rewriting the
original decision criteria.

## Research question

Does claim-based selection still lose policy-quality ranking when candidate-defined
reward shaping is removed from the training fitness? This tests whether the protocol-v3
failure needs candidate-controlled shaping, versus remaining measurement choices
(training seeds, episode count, and horizon). It is an **extreme ablation**, not a
clean isolation of reward *scale* alone: removing shaping also changes its content.
A bounded, candidate-controlled shaping condition would be needed to separate scale
from other shaping effects.

## Existing evidence and its limit

The protocol-v3 LunarLander study has three independent outer loops per selection
mode. Candidate-shaped training rewards may reach absolute value 1e6 per transition.
In all three claim-selected runs, the maximum claim reached 1e9.

As a **post-hoc diagnostic only**, selecting the maximum archived `train_raw_of_best`
within each existing claim-selected archive yields the following raw-audit regrets:

| Outer seed | Actual claim-selected regret | Post-hoc raw-training pick regret |
| ---: | ---: | ---: |
| 101 | 990.8 | 198.1 |
| 202 | 337.3 | 131.7 |
| 303 | 802.9 | 0.0 |

This is not a counterfactual outer run. Changing the selection signal would change
parent choices, LLM feedback, and the candidates generated later.

## Prospective control protocol

Use a separate, versioned protocol. The new factor is **training reward source**:

- `candidate_shaped`: current protocol-v3 behavior; candidate `shaped_reward`
  determines training fitness.
- `fixed_raw`: the harness uses only environment raw reward for training fitness;
  candidate `shaped_reward` remains syntactically valid but is not executed during
  training. The LLM prompt must state this clearly.

Cross this factor with the existing selection factor (`audited`, `claimed`). Keep
LunarLander, policy architecture, CMA-ES trainer, candidate count (15), training
budget (2,000 episodes and 600 seconds per candidate), 32-episode fresh-seed
raw-reward audit, and parent-selection rule fixed. The `fixed_raw` condition still
lets recipes change optimizer settings and evaluation schedule. Consequently, a
remaining claim/audit gap can arise from training-seed reuse, sampling noise, and
short horizons; it must not automatically be called reward-scale gaming.

Run at least three independent outer loops per cell (2 reward sources x 2 selection
rules x 3 seeds = 12 runs, 180 candidates) as an initial, descriptive study. Because
the mutation service can change over time, run all four cells contemporaneously or
describe comparisons with archived protocol-v3 runs as time-confounded. Reusing
outer RNG seed labels does not make LLM outputs and audit seeds strictly paired.

## Predeclared outcomes and interpretation

Primary: raw-audit return of the candidate actually selected at the end of each
outer run. Secondary: within-archive selection regret (best audited candidate minus
selected candidate), number of maximum-claim ties, claim distribution, and the
audit quality of the post-hoc archive oracle. A zero regret under audit selection
is true by construction and is not an empirical success metric.

Interpretation:

- If `fixed_raw` removes score saturation and materially reduces claim-selected
  regret, removing candidate-controlled shaping is an effective mitigation **in
  this setup**. This does not establish that scale alone caused the difference.
- If `fixed_raw` removes saturation but large regret remains, other candidate-
  controlled measurement choices warrant separate controls.
- If the result varies across outer runs, report that variation rather than
  declaring a general rule from a favorable subset.

Do not infer generality across environments, models, or recipe retraining from
this one-environment control. Do not pool candidates or audit episodes as
independent outer-loop replicates. Do not use a post-hoc significance test to
retroactively select the primary outcome.

## Execution gate and estimated resources

Completed preflight: `--training-reward fixed_raw` is a separate protocol-v4
prompt profile and an immutable run-config field. The default `candidate_shaped`
profile retains the protocol-v3 prompt hash and old run-config shape. The fixed-raw
adapter substitutes a harness-owned raw-reward function during training; no
candidate shaping executes there. Twenty-three local tests pass, and one
fixed-raw LunarLander seed-only smoke completed with no model call. A full
four-cell experiment has **not** been run.

Before any LLM run: inspect the exact commands and fresh run roots. Start with
one pilot outer run per fixed-raw selection condition only if the user approves
the model-call/compute budget. Do not treat the pilot as confirmatory evidence.

## First approved pilot (2026-09-21)

One `fixed_raw` + `claimed` LunarLander outer run completed in
`runs/protocol_v4_fixed_raw/pilot_claimed_seed101_retry` (outer seed 101). It has
15 completed candidates, zero failed records, 14 Codex requests, zero rejected
attempts, 249,584 recorded input tokens, and 3,999 output tokens. Every candidate's
claim equals its archived `train_raw_of_best`; each has 32 audit seeds and returns.
The selected candidate (id 10) claimed 296.34 and audited -166.91. The archive's
post-hoc audit oracle (id 8) audited 5.39, giving 172.30 within-archive regret.
Archive SHA-256: `b77233c087a181c5b1c0882980fc2b47411680e2a53422d416644e9f664cc873`.
This single pilot is **not** confirmatory evidence and is not pooled with
protocol-v3 results. A first sandboxed attempt in a separate directory stopped
at the first mutation because the Codex CLI could not write its local database;
it is excluded from the successful pilot.

The second approved pilot used `fixed_raw` + `audited` with the same outer RNG
seed 101 and completed in `runs/protocol_v4_fixed_raw/pilot_audited_seed101`.
It has 15 completed candidates, zero failed records, 14 Codex requests, zero
rejected attempts, 249,565 recorded input tokens, and 5,562 output tokens.
The selected candidate (id 13) claimed 250.58 and audited -6.67; it is also
the within-archive audit oracle, so regret is zero by construction. Archive
SHA-256: `dee7a6d8a607f4fce3bed095e143390ad3281e2733feb9e4c54513c27fcb5998`.
The two pilot final-pick audits differ by 160.24 raw-reward points, but this is
one independent outer run per mode, not a strictly paired comparison or an
estimate of expected benefit. Both runs recorded the same protocol-v4 prompt
hash `f420fda49b01c141784c78adda11992bd624102a3c93c1960019891e96e107df`.

## Completed three-run-per-mode ablation (2026-09-23)

Seeds 202 and 303 were subsequently completed for both selection modes, giving
three independent protocol-v4 outer runs per mode and 90 candidates total. All
six runs have 15 completed candidates, zero failed records, 32 audit episodes
per candidate, the same prompt hash and mutator identity, and claims equal to
the archived raw training fitness. The machine-readable summary is
`evidence/protocol_v4_fixed_raw_summary.json`, reconstructed by
`scripts/analyze_fixed_raw_replications.py` without executing generated recipes.

| Seed | Audited-mode final audit | Claimed-mode final audit | Claimed-run oracle | Claimed regret |
| ---: | ---: | ---: | ---: | ---: |
| 101 | -6.67 | -166.91 | 5.39 | 172.30 |
| 202 | -69.29 | -140.11 | -54.14 | 85.97 |
| 303 | 17.90 | -104.21 | -30.29 | 73.93 |
| Mean | -19.36 | -137.08 | -26.34 | 110.73 |

The seed-aligned audited-minus-claimed final-pick differences are 160.24,
70.82, and 122.11 raw-reward points (mean 117.72). Claim selection missed a
better audited candidate already present in its own archive in all three runs.
The selected claimed-mode candidates used one training episode per evaluation
at the generation that produced their maximum raw-training score. This is a
post-hoc schedule observation, not an isolated causal mechanism.

Across all six runs the mutation operator made 84 requests, with zero rejected
attempts, 1,498,354 recorded input tokens, and 30,339 output tokens. The result
remains a descriptive, one-environment ablation with n=3 independent outer runs
per mode. It supports the narrower conclusion that removing candidate-defined
reward shaping eliminates the 1e9 scale saturation and greatly reduces the
observed claim-selection regret relative to protocol v3, but does not remove
within-archive mis-ranking in these runs. Cross-protocol effect-size comparisons
are time-confounded and are not randomized paired estimates.

The six archived runs took roughly 30 minutes wall time in total and made 84
Codex requests (1.49 million recorded input tokens, 46,924 output tokens).
A fresh 12-run factorial at the same candidate count would be approximately
60 minutes and 168 requests, with roughly 3 million input and 94,000 output
tokens if behavior matches history. These are planning estimates, not guaranteed
runtime or billing. No full experiment starts without explicit approval of its
compute/model-call budget. Stop on provider quota/auth failure.
