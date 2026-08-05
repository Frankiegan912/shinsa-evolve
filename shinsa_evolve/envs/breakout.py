"""MinAtar Breakout adapter on gymnax: jit-compiled, vmapped batch rollouts.

Rollout length is quantized to a few fixed buckets so jit compiles at most
len(STEP_BUCKETS) variants; outputs are truncated to the requested max_steps.
gymnax auto-resets on episode end, so everything after the first done is masked
out via the alive flags recorded during the scan.
"""

from __future__ import annotations

import numpy as np

from ..core.policy import DEFAULT_HIDDEN, layer_shapes
from ..core.shaping import shaped_return

STEP_BUCKETS = (128, 256, 512, 1000)
CHUNK = 256  # rollouts per compiled batch call, to bound device memory


def _bucket(max_steps: int) -> int:
    for b in STEP_BUCKETS:
        if max_steps <= b:
            return b
    return STEP_BUCKETS[-1]


class BreakoutAdapter:
    name = "breakout"

    def __init__(self, hidden: tuple[int, ...] = DEFAULT_HIDDEN):
        import jax  # local import: lunarlander-only runs never need jax
        import gymnax

        self._jax = jax
        self.env, self.env_params = gymnax.make("Breakout-MinAtar")
        obs_shape = self.env.observation_space(self.env_params).shape
        self.obs_dim = int(np.prod(obs_shape))
        self.n_actions = int(self.env.num_actions)
        self.sizes = (self.obs_dim, *hidden, self.n_actions)
        self.audit_max_steps = 1000
        self._compiled = {}

    # -- jitted rollout -------------------------------------------------------

    def _rollout_fn(self, T: int):
        if T in self._compiled:
            return self._compiled[T]
        jax = self._jax
        import jax.numpy as jnp

        env, env_params = self.env, self.env_params
        shapes = layer_shapes(self.sizes)
        n_layers = len(shapes)
        # Precompute static slice offsets for unflattening inside jit.
        offsets = []
        i = 0
        for nin, nout in shapes:
            offsets.append((i, i + nin * nout, i + nin * nout + nout, nin, nout))
            i += nin * nout + nout

        def forward(flat, x):
            for k, (w0, w1, b1, nin, nout) in enumerate(offsets):
                w = flat[w0:w1].reshape(nin, nout)
                b = flat[w1:b1]
                x = x @ w + b
                if k < n_layers - 1:
                    x = jnp.tanh(x)
            return x

        def single(flat, seed):
            key = jax.random.PRNGKey(seed)
            key, kr = jax.random.split(key)
            obs, state = env.reset(kr, env_params)

            def step(carry, _):
                obs, state, key, done = carry
                obs_flat = obs.reshape(-1).astype(jnp.float32)
                action = jnp.argmax(forward(flat, obs_flat))
                key, ks = jax.random.split(key)
                nobs, nstate, reward, ndone, _ = env.step(ks, state, action, env_params)
                alive = jnp.logical_not(done)
                out = (
                    obs.reshape(-1).astype(jnp.uint8),
                    action.astype(jnp.int32),
                    (reward * alive).astype(jnp.float32),
                    alive,
                    jnp.logical_and(alive, ndone),  # natural termination at this step
                )
                return (nobs, nstate, key, jnp.logical_or(done, ndone)), out

            init = (obs, state, jnp.asarray(key), jnp.asarray(False))
            _, (obs_seq, act_seq, rew_seq, alive_seq, term_seq) = jax.lax.scan(
                step, init, None, length=T
            )
            return obs_seq, act_seq, rew_seq, alive_seq, term_seq

        fn = jax.jit(jax.vmap(single, in_axes=(0, 0)))
        self._compiled[T] = fn
        return fn

    def _run_batch(self, params_2d: np.ndarray, seeds: np.ndarray, T: int):
        """Run N rollouts of bucket length T; returns numpy arrays (N,T,...)."""
        jax = self._jax
        fn = self._rollout_fn(T)
        outs = []
        for lo in range(0, len(seeds), CHUNK):
            hi = min(lo + CHUNK, len(seeds))
            chunk = fn(
                params_2d[lo:hi].astype(np.float32),
                seeds[lo:hi].astype(np.int32),
            )
            outs.append([np.asarray(jax.device_get(a)) for a in chunk])
        return [np.concatenate([o[k] for o in outs], axis=0) for k in range(5)]

    # -- adapter API ----------------------------------------------------------

    def eval_shaped(self, members, seeds, max_steps: int, recipe):
        P, E = len(members), len(seeds)
        T = _bucket(max_steps)
        params_2d = np.repeat(np.stack(members), E, axis=0)  # (P*E, dim)
        seed_arr = np.tile(np.asarray(seeds), P)
        obs, act, rew, alive, term = self._run_batch(params_2d, seed_arr, T)

        ms = min(max_steps, T)
        shaped = np.zeros((P, E))
        raw = np.zeros((P, E))
        for n in range(P * E):
            p, e = divmod(n, E)
            a = alive[n, :ms]
            L = int(a.sum())
            if L == 0:
                continue
            terminated = bool(term[n, :L][-1])
            shaped[p, e] = shaped_return(
                recipe, obs[n, :L].astype(np.float32), act[n, :L], rew[n, :L], terminated
            )
            raw[p, e] = float(rew[n, :L].sum())
        return shaped, raw

    def evaluate_raw(self, params, seeds, max_steps: int):
        T = _bucket(max_steps)
        params_2d = np.repeat(np.asarray(params)[None, :], len(seeds), axis=0)
        _, _, rew, alive, _ = self._run_batch(params_2d, np.asarray(seeds), T)
        ms = min(max_steps, T)
        return (rew[:, :ms] * alive[:, :ms]).sum(axis=1)

    def close(self):
        pass
