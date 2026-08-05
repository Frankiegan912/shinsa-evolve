# Paper outline (arXiv short paper, 6-8 pages) — draft target 8/18-8/24

Working title: "Audit-Gated Recipe Evolution: Making Self-Reported Results from
LLM-Driven Program Search Trustworthy"

1. Introduction — LLM-driven program search selects on self-reported fitness; when the
   candidate controls its own measurement, selection can fail silently.
2. System — recipe contract; inner training (fixed MLP + SepCMA-ES, hard budgets);
   the audit gate (fresh OS-entropy seeds, raw return, no candidate influence);
   dual-score archive.
3. Experimental setup — Breakout (gymnax/jit) + LunarLander (gymnasium/spawn pool);
   audited vs claimed selection; matrix 60/20/40/15; seed-recipe variance baseline (3x).
4. Results — honesty gap distribution (one-sidedness); selection trajectories;
   claimed-mode selection failure (picked-by-claimed vs best-available); budget-aligned
   truncated comparison.
5. Limitations — audit certifies policies, not recipes (variance baseline quantifies);
   hygiene checks are not a sandbox; two environments only.
6. Related work + conclusion.
