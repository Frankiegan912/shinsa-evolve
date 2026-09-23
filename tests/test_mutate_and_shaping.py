import os
import stat
import subprocess
from pathlib import Path

import numpy as np

from shinsa_evolve.core import llm_mutate
from shinsa_evolve.core.orchestrate import FixedRawTrainingAdapter
from shinsa_evolve.core.interface import load_recipe
from shinsa_evolve.core.shaping import shaped_return

REPO = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).parent / "fake_claude.sh"


def test_extract_code_picks_last_fence():
    text = "```python\nx = 1\n```\nmore\n```python\ny = 2\n```"
    assert llm_mutate.extract_code(text) == "y = 2\n"
    assert llm_mutate.extract_code("no fences here") is None


def test_mutate_via_fake_cli(tmp_path, monkeypatch):
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("SHINSA_CLAUDE_BIN", str(FAKE))
    parent = (REPO / "recipes" / "seed_lunarlander.py").read_text()
    out = tmp_path / "child.py"
    source, attempts = llm_mutate.mutate(parent, 10.0, "test env", "audited", 8, out)
    assert out.exists()
    assert attempts[-1]["ok"] is True
    module = load_recipe(out)
    assert module.OPTIMIZER["popsize"] == 12


def test_mutate_via_codex_records_usage(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["codex", "exec"]
        assert "--ephemeral" in cmd
        assert "--ignore-user-config" in cmd
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"
        output = Path(cmd[cmd.index("-o") + 1])
        output.write_text(
            "```python\n"
            "OPTIMIZER = {'sigma0': 0.4, 'popsize': 12}\n"
            "def schedule(gen):\n"
            "    return {'episodes_per_eval': 6, 'max_steps': 400}\n"
            "def shaped_reward(obs, action, reward, terminated, truncated, step):\n"
            "    return reward\n"
            "```\n"
        )
        stdout = '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}\n'
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setenv("SHINSA_MUTATE_PROVIDER", "codex")
    monkeypatch.setenv("SHINSA_MUTATE_MODEL", "test-codex")
    monkeypatch.setattr(llm_mutate.subprocess, "run", fake_run)
    parent = (REPO / "recipes" / "seed_lunarlander.py").read_text()
    out = tmp_path / "child.py"
    _, attempts = llm_mutate.mutate(parent, 10.0, "test env", "audited", 8, out)
    assert attempts[-1]["ok"] is True
    assert attempts[-1]["usage"] == {"input_tokens": 100, "output_tokens": 20}


def test_codex_quota_error_is_not_retried(tmp_path, monkeypatch):
    calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="usage limit reached")

    monkeypatch.setenv("SHINSA_MUTATE_PROVIDER", "codex")
    monkeypatch.setenv("SHINSA_MUTATE_MODEL", "test-codex")
    monkeypatch.setattr(llm_mutate.subprocess, "run", fake_run)
    parent = (REPO / "recipes" / "seed_lunarlander.py").read_text()
    with np.testing.assert_raises(llm_mutate.ProviderUnavailable):
        llm_mutate.mutate(
            parent, 10.0, "test env", "audited", 8, tmp_path / "child.py",
        )
    assert calls == 1


def test_rejection_reason_is_fed_back_and_attempts_stay_structured(tmp_path, monkeypatch):
    prompts = []
    replies = iter([
        "```python\n"
        "OPTIMIZER = {'sigma0': 0.4, 'popsize': 12}\n"
        "def schedule(gen): return {'episodes_per_eval': 6, 'max_steps': 400}\n"
        "def shaped_reward(obs, action, reward, terminated, truncated, step): return 1e9\n"
        "```",
        "```python\n"
        "OPTIMIZER = {'sigma0': 0.4, 'popsize': 12}\n"
        "def schedule(gen): return {'episodes_per_eval': 6, 'max_steps': 400}\n"
        "def shaped_reward(obs, action, reward, terminated, truncated, step): return reward\n"
        "```",
    ])

    def fake_call(prompt, timeout=300.0):
        prompts.append(prompt)
        return next(replies), {}

    monkeypatch.setattr(llm_mutate, "call_model", fake_call)
    parent = (REPO / "recipes" / "seed_lunarlander.py").read_text()
    _, attempts = llm_mutate.mutate(
        parent, 999_996.0, "test env", "claimed", 8, tmp_path / "child.py",
    )
    assert len(attempts) == 2
    assert attempts[0]["stage"] == "validate"
    assert "oversized value" in prompts[1]
    assert llm_mutate.mutation_prompt_sha256() == llm_mutate.mutation_prompt_sha256()


def test_shaped_return_matches_hand_computation():
    class R:
        @staticmethod
        def shaped_reward(obs, action, reward, terminated, truncated, step):
            return reward * 2 + (5.0 if terminated else 0.0) + (1.0 if truncated else 0.0)

    obs = np.zeros((3, 4), dtype=np.float32)
    act = np.array([0, 1, 0])
    rew = np.array([1.0, 0.0, 2.0])
    # terminated episode: 2*(1+0+2) + 5 at the last step
    assert shaped_return(R, obs, act, rew, terminated=True) == 11.0
    # truncated episode: 2*3 + 1
    assert shaped_return(R, obs, act, rew, terminated=False) == 7.0


def test_shaped_return_guards_nonfinite():
    class R:
        @staticmethod
        def shaped_reward(obs, action, reward, terminated, truncated, step):
            return float("inf")

    obs = np.zeros((2, 4), dtype=np.float32)
    assert shaped_return(R, obs, np.zeros(2, int), np.zeros(2), False) == -1e9


def test_fixed_raw_training_ignores_candidate_shaping():
    class Candidate:
        @staticmethod
        def shaped_reward(*args):
            raise AssertionError("candidate shaping must not execute")

    class Base:
        sizes = (8, 32, 32, 4)
        obs_dim = 8

        def eval_shaped(self, members, seeds, max_steps, recipe):
            obs = np.zeros((2, 8), dtype=np.float32)
            actions = np.array([0, 1])
            rewards = np.array([1.0, 2.0])
            value = shaped_return(recipe, obs, actions, rewards, True)
            return np.array([[value]]), np.array([[rewards.sum()]])

        def evaluate_raw(self, params, seeds, max_steps):
            return np.array([3.0])

        def close(self):
            pass

    adapter = FixedRawTrainingAdapter(Base())
    shaped, raw = adapter.eval_shaped([np.zeros(1)], [1], 2, Candidate())
    np.testing.assert_array_equal(shaped, [[3.0]])
    np.testing.assert_array_equal(raw, [[3.0]])
    np.testing.assert_array_equal(adapter.evaluate_raw(None, [1], 2), [3.0])


def test_fixed_raw_prompt_has_distinct_provenance():
    assert llm_mutate.mutation_prompt_sha256() == (
        "be9c5020a26d3bae12e5ec8ec5f2e3b2febfa34d15ab2d9583ede47289a00e8a"
    )
    assert llm_mutate.mutation_prompt_sha256("fixed_raw") != llm_mutate.mutation_prompt_sha256()
    prompt = llm_mutate.build_prompt("pass", 0.0, "LunarLander", "claimed", "fixed_raw")
    assert "shaped_reward function is still required" in prompt
    assert "not executed during training or audit" in prompt
