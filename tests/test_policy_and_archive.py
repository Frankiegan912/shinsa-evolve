import numpy as np
import pytest

from shinsa_evolve.core import orchestrate
from shinsa_evolve.core.archive import Archive
from shinsa_evolve.core.orchestrate import (
    ensure_run_config,
    load_outer_rng,
    save_outer_rng,
    select_parent,
)
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


def test_run_config_is_immutable_on_resume(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {"schema_version": 2, "env": "breakout", "rng_seed": 1}
    ensure_run_config(run_dir, config)
    ensure_run_config(run_dir, dict(config))
    with pytest.raises(RuntimeError, match="changed config keys: \\['rng_seed'\\]"):
        ensure_run_config(run_dir, {**config, "rng_seed": 2})


def test_outer_rng_checkpoint_preserves_parent_sequence(tmp_path):
    archive = Archive(tmp_path / "run")
    for cid in range(6):
        archive.append({
            "id": cid,
            "status": "done",
            "parent": None,
            "claimed": float(cid),
            "audited": float(10 - cid),
        })

    uninterrupted = load_outer_rng(archive.run_dir, 7, archive)
    expected = [select_parent(archive, "audited", uninterrupted)["id"] for _ in range(12)]

    state_path = archive.run_dir / "outer_rng_state.json"
    state_path.unlink()
    first_process = load_outer_rng(archive.run_dir, 7, archive)
    observed = []
    for _ in range(5):
        observed.append(select_parent(archive, "audited", first_process)["id"])
        save_outer_rng(archive.run_dir, 7, first_process)
    resumed = load_outer_rng(archive.run_dir, 7, archive)
    for _ in range(7):
        observed.append(select_parent(archive, "audited", resumed)["id"])
        save_outer_rng(archive.run_dir, 7, resumed)
    assert observed == expected


def test_outer_rng_refuses_ambiguous_legacy_resume(tmp_path):
    archive = Archive(tmp_path / "run")
    archive.append({
        "id": 0,
        "status": "done",
        "parent": None,
        "claimed": 0.0,
        "audited": 0.0,
    })
    archive.append({
        "id": 1,
        "status": "done",
        "parent": 0,
        "claimed": 1.0,
        "audited": 1.0,
    })
    with pytest.raises(RuntimeError, match="RNG state is missing"):
        load_outer_rng(archive.run_dir, 1, archive)


def test_interrupted_main_preserves_parent_sequence(tmp_path, monkeypatch):
    class DummyAdapter:
        obs_dim = 8

        def close(self):
            pass

    monkeypatch.setattr(orchestrate, "get_adapter", lambda *args, **kwargs: DummyAdapter())
    monkeypatch.setattr(
        orchestrate.llm_mutate,
        "mutator_identity",
        lambda: {"binary": "fake", "cli_version": "test", "model": "test-model"},
    )

    def fake_mutate(parent_source, score, env_desc, mode, obs_dim, out_path):
        out_path.write_text(parent_source)
        return parent_source, [{"attempt": 0, "ok": True, "seconds": 0.0}]

    monkeypatch.setattr(orchestrate.llm_mutate, "mutate", fake_mutate)

    class InjectedStop(BaseException):
        pass

    def processor(stop_after=None):
        def fake_process(cid, recipe_path, parent_id, mutate_log, adapter, budget,
                         audit_episodes, archive, mode, env_name, train_seed):
            archive.append({
                "id": cid,
                "parent": parent_id,
                "status": "done",
                "claimed": float(cid),
                "audited": float(cid),
                "recipe": str(recipe_path.relative_to(archive.run_dir)),
            })
            if stop_after is not None and archive.count_done() == stop_after:
                raise InjectedStop()
        return fake_process

    common = ["--env", "lunarlander", "--mode", "audited", "--candidates", "8"]
    continuous_dir = tmp_path / "continuous"
    monkeypatch.setattr(orchestrate, "process_candidate", processor())
    assert orchestrate.main([*common, "--run-dir", str(continuous_dir)]) == 0

    interrupted_dir = tmp_path / "interrupted"
    monkeypatch.setattr(orchestrate, "process_candidate", processor(stop_after=4))
    with pytest.raises(InjectedStop):
        orchestrate.main([*common, "--run-dir", str(interrupted_dir)])
    monkeypatch.setattr(orchestrate, "process_candidate", processor())
    assert orchestrate.main([*common, "--run-dir", str(interrupted_dir)]) == 0

    continuous_parents = [r["parent"] for r in Archive(continuous_dir).done()]
    interrupted_parents = [r["parent"] for r in Archive(interrupted_dir).done()]
    assert interrupted_parents == continuous_parents


def test_llm_run_requires_explicit_model(monkeypatch):
    monkeypatch.delenv("SHINSA_MUTATE_MODEL", raising=False)
    with pytest.raises(orchestrate.llm_mutate.MutateError, match="must be set explicitly"):
        orchestrate.llm_mutate.mutator_identity()
