from pathlib import Path

from shinsa_evolve.core.interface import (
    clamp_schedule,
    load_recipe,
    static_check,
    validate_recipe,
)

REPO = Path(__file__).resolve().parents[1]


def test_seed_recipes_validate():
    for name, obs_dim in (("seed_breakout.py", 400), ("seed_lunarlander.py", 8)):
        module = load_recipe(REPO / "recipes" / name)
        assert validate_recipe(module, obs_dim) == []


def test_static_check_rejects_bad_imports():
    assert static_check("import os\n")
    assert static_check("from subprocess import run\n")
    assert static_check("open('/etc/passwd')\n")
    assert static_check("x.__globals__\n")
    assert not static_check("import math\nimport numpy as np\n")


def test_validate_rejects_broken_contract():
    src = "OPTIMIZER = {'sigma0': 99.0, 'popsize': 2}\n"
    p = Path(__file__).parent / "_tmp_bad.py"
    p.write_text(src)
    try:
        module = load_recipe(p)
        problems = validate_recipe(module, 8)
        assert any("sigma0" in x for x in problems)
        assert any("popsize" in x for x in problems)
        assert any("schedule" in x for x in problems)
        assert any("shaped_reward" in x for x in problems)
    finally:
        p.unlink()


def test_validate_rejects_nondeterministic_shaping():
    src = (
        "import numpy as np\n"
        "OPTIMIZER = {'sigma0': 0.5, 'popsize': 8}\n"
        "def schedule(gen):\n"
        "    return {'episodes_per_eval': 4, 'max_steps': 500}\n"
        "def shaped_reward(obs, action, reward, terminated, truncated, step):\n"
        "    return reward + np.random.rand()\n"
    )
    p = Path(__file__).parent / "_tmp_rand.py"
    p.write_text(src)
    try:
        module = load_recipe(p)
        assert any("deterministic" in x for x in validate_recipe(module, 8))
    finally:
        p.unlink()


def test_clamp_schedule():
    assert clamp_schedule({"episodes_per_eval": 999, "max_steps": 5}) == (16, 50)
    assert clamp_schedule({}) == (4, 500)
