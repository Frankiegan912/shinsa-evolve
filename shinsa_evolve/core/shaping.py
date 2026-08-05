"""Apply a recipe's shaped_reward over a recorded trajectory.

Both environment adapters return raw per-step trajectories; shaping is always
applied here, in harness code, so the contract semantics are identical across
execution backends (jit batch rollouts vs process pools).
"""

from __future__ import annotations

import numpy as np


def shaped_return(recipe, obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                  terminated: bool) -> float:
    """Sum shaped_reward over one episode.

    obs: (L, obs_dim) float32; act: (L,) int; rew: (L,) float; terminated: True if
    the episode ended naturally at step L-1 (otherwise the final step is truncation).
    """
    L = len(rew)
    total = 0.0
    for t in range(L):
        last = t == L - 1
        total += float(recipe.shaped_reward(
            obs[t], int(act[t]), float(rew[t]),
            bool(last and terminated), bool(last and not terminated), t,
        ))
        if not np.isfinite(total):
            return -1e9
    return total
