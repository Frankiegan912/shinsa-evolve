import numpy as np
import pytest

from shinsa_evolve.core.archive import Archive
from shinsa_evolve.core.policy import act_np, forward_np, param_count


def test_param_count_and_forward_shapes():
    sizes = (8, 32, 32, 4)
    dim = param_count(sizes)
    assert dim == 8 * 32 + 32 + 32 * 32 + 32 + 32 * 4 + 4
    flat = np.random.default_rng(0).normal(size=dim)
    logits = forward_np(flat, np.ones(8), sizes)
    assert logits.shape == (4,)
    assert 0 <= act_np(flat, np.ones(8), sizes) < 4


def test_forward_parity_numpy_vs_jax():
    jax = pytest.importorskip("jax")
    from shinsa_evolve.envs.breakout import BreakoutAdapter

    adapter = BreakoutAdapter()
    sizes = adapter.sizes
    rng = np.random.default_rng(1)
    flat = rng.normal(size=param_count(sizes)).astype(np.float32)
    obs = rng.random(sizes[0]).astype(np.float32)

    # Rebuild the jitted forward exactly as the rollout uses it.
    import jax.numpy as jnp
    from shinsa_evolve.core.policy import layer_shapes

    shapes = layer_shapes(sizes)
    offsets, i = [], 0
    for nin, nout in shapes:
        offsets.append((i, i + nin * nout, i + nin * nout + nout, nin, nout))
        i += nin * nout + nout

    x = jnp.asarray(obs)
    for k, (w0, w1, b1, nin, nout) in enumerate(offsets):
        w = jnp.asarray(flat[w0:w1]).reshape(nin, nout)
        b = jnp.asarray(flat[w1:b1])
        x = x @ w + b
        if k < len(shapes) - 1:
            x = jnp.tanh(x)

    np.testing.assert_allclose(np.asarray(x), forward_np(flat, obs, sizes), rtol=1e-4, atol=1e-4)


def test_archive_roundtrip_and_ranking(tmp_path):
    a = Archive(tmp_path / "run")
    a.append({"id": 0, "status": "done", "claimed": 1.0, "audited": 5.0})
    a.append({"id": 1, "status": "done", "claimed": 9.0, "audited": 2.0})
    a.append({"id": 2, "status": "mutate_failed"})
    assert a.count_done() == 2
    assert a.next_id() == 3
    assert a.top("audited", 1)[0]["id"] == 0
    assert a.top("claimed", 1)[0]["id"] == 1


def test_archive_tolerates_partial_line(tmp_path):
    a = Archive(tmp_path / "run")
    a.append({"id": 0, "status": "done", "claimed": 1.0, "audited": 1.0})
    with open(a.path, "a") as f:
        f.write('{"id": 1, "status": "do')  # simulated kill mid-write
    assert a.count_done() == 1
    assert a.next_id() == 1
