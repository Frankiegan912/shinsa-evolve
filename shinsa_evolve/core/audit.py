"""The audit gate: candidate-independent evaluation of a trained policy.

Three invariants:
  1. Seeds are freshly drawn from OS entropy at audit time; no candidate can
     predict or overfit them.
  2. Scoring uses the environment's raw reward only; shaped_reward is never
     imported or called here.
  3. Recipe code has no execution path into this module; the audit sees only the
     trained parameter vector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class AuditResult:
    score: float
    std: float
    per_episode: list
    seeds: list

    def stats(self) -> dict:
        return {"audited_score": self.score, "audited_std": self.std, "episodes": len(self.per_episode)}


def fresh_seeds(n: int) -> list[int]:
    return [int.from_bytes(os.urandom(4), "little") % (2**31 - 1) for _ in range(n)]


def audit(params: np.ndarray, adapter, episodes: int) -> AuditResult:
    seeds = fresh_seeds(episodes)
    raw = adapter.evaluate_raw(params, seeds, adapter.audit_max_steps)  # (E,)
    raw = np.asarray(raw, dtype=np.float64)
    return AuditResult(
        score=float(raw.mean()),
        std=float(raw.std()),
        per_episode=[float(x) for x in raw],
        seeds=seeds,
    )
