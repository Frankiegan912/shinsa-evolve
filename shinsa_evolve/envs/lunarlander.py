"""LunarLander-v3 adapter on gymnasium: spawn-based process pool.

Workers run whole episodes with the numpy policy and return raw trajectories;
shaping is applied in the parent via core.shaping so contract semantics match the
jit backend exactly. Audit calls use the same workers but only raw returns.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from ..core.policy import DEFAULT_HIDDEN, act_np
from ..core.shaping import shaped_return

_SIZES = (8, *DEFAULT_HIDDEN, 4)


def _run_episode(payload):
    flat, seed, max_steps = payload
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    try:
        obs, _ = env.reset(seed=int(seed))
        obs_l, act_l, rew_l = [], [], []
        terminated = False
        for _ in range(max_steps):
            action = act_np(flat, obs, _SIZES)
            nobs, reward, term, trunc, _ = env.step(action)
            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(action)
            rew_l.append(float(reward))
            obs = nobs
            if term or trunc:
                terminated = bool(term)
                break
        return (
            np.asarray(obs_l, dtype=np.float32),
            np.asarray(act_l, dtype=np.int32),
            np.asarray(rew_l, dtype=np.float64),
            terminated,
        )
    finally:
        env.close()


class LunarLanderAdapter:
    name = "lunarlander"

    def __init__(self, pool_size: int | None = None):
        self.sizes = _SIZES
        self.obs_dim = 8
        self.n_actions = 4
        self.audit_max_steps = 1000
        self.pool_size = pool_size or max(2, (os.cpu_count() or 4) - 2)
        self._pool = None

    def _pool_or_start(self):
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(processes=self.pool_size)
        return self._pool

    def _episodes(self, params_list, seeds, max_steps: int):
        tasks = [
            (np.asarray(flat, dtype=np.float32), seed, int(max_steps))
            for flat in params_list
            for seed in seeds
        ]
        return self._pool_or_start().map(_run_episode, tasks)

    # -- adapter API ----------------------------------------------------------

    def eval_shaped(self, members, seeds, max_steps: int, recipe):
        P, E = len(members), len(seeds)
        results = self._episodes(members, seeds, max_steps)
        shaped = np.zeros((P, E))
        raw = np.zeros((P, E))
        for n, (obs, act, rew, terminated) in enumerate(results):
            p, e = divmod(n, E)
            if len(rew) == 0:
                continue
            shaped[p, e] = shaped_return(recipe, obs, act, rew, terminated)
            raw[p, e] = float(rew.sum())
        return shaped, raw

    def evaluate_raw(self, params, seeds, max_steps: int):
        results = self._episodes([params], seeds, max_steps)
        return np.asarray([float(r[2].sum()) for r in results])

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
