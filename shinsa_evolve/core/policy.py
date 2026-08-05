"""Fixed policy: a small tanh MLP over a flat parameter vector (numpy reference).

The architecture is harness-fixed per environment; recipes cannot change it.
The JAX forward pass in envs/breakout.py must stay numerically equivalent to
forward_np (covered by a parity test).
"""

from __future__ import annotations

import numpy as np

DEFAULT_HIDDEN = (32, 32)


def layer_shapes(sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def param_count(sizes: tuple[int, ...]) -> int:
    return sum(nin * nout + nout for nin, nout in layer_shapes(sizes))


def unflatten(flat: np.ndarray, sizes: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    i = 0
    for nin, nout in layer_shapes(sizes):
        w = flat[i:i + nin * nout].reshape(nin, nout)
        i += nin * nout
        b = flat[i:i + nout]
        i += nout
        layers.append((w, b))
    return layers


def forward_np(flat: np.ndarray, obs: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    layers = unflatten(np.asarray(flat, dtype=np.float32), sizes)
    x = np.asarray(obs, dtype=np.float32)
    for k, (w, b) in enumerate(layers):
        x = x @ w + b
        if k < len(layers) - 1:
            x = np.tanh(x)
    return x


def act_np(flat: np.ndarray, obs: np.ndarray, sizes: tuple[int, ...]) -> int:
    return int(np.argmax(forward_np(flat, obs, sizes)))
