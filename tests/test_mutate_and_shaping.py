import os
import stat
from pathlib import Path

import numpy as np

from shinsa_evolve.core import llm_mutate
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
