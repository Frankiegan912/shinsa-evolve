"""Recipe contract: what a candidate recipe may control, and how it is validated.

A *recipe* is a small Python module written by the LLM mutation operator (or by a
human, for seeds). It controls exactly three things about inner training:

  OPTIMIZER      dict: {"sigma0": float, "popsize": int} for separable CMA-ES
  schedule(gen)  -> {"episodes_per_eval": int, "max_steps": int} for generation gen
  shaped_reward(obs, action, reward, terminated, truncated, step) -> float
                 per-transition training reward (the candidate-visible fitness signal)

Everything else (policy architecture, training loop, hard budgets, the audit) is
fixed harness code that recipe code never touches. The static checks below are
hygiene against accidental IO/imports in generated code, not a security sandbox.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np

SIGMA0_RANGE = (0.01, 2.0)
POPSIZE_RANGE = (4, 64)
EPISODES_RANGE = (1, 16)
MAX_STEPS_RANGE = (50, 1000)
SHAPED_REWARD_ABS_MAX = 1e6

ALLOWED_IMPORTS = {"math", "numpy"}
FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}

CONTRACT_SPEC = f"""A recipe is a single Python module that defines exactly:

1) OPTIMIZER: dict
   - "sigma0": float in [{SIGMA0_RANGE[0]}, {SIGMA0_RANGE[1]}]   (initial CMA-ES step size)
   - "popsize": int in [{POPSIZE_RANGE[0]}, {POPSIZE_RANGE[1]}]        (CMA-ES population per generation)

2) def schedule(gen: int) -> dict
   Per-generation training settings:
   - "episodes_per_eval": int in [{EPISODES_RANGE[0]}, {EPISODES_RANGE[1]}]  (episodes averaged per candidate evaluation)
   - "max_steps": int in [{MAX_STEPS_RANGE[0]}, {MAX_STEPS_RANGE[1]}]       (episode step cap during training)

3) def shaped_reward(obs, action, reward, terminated, truncated, step) -> float
   Per-transition training reward. Inputs: obs (1-D float numpy array), action (int),
   reward (float, the environment's raw reward for this step), terminated (bool),
   truncated (bool), step (int, 0-based step index). Must be deterministic, finite, and
   have absolute value <= {SHAPED_REWARD_ABS_MAX:g} for every transition.

Only `math` and `numpy` may be imported. Total training compute is capped by a hard
budget (episodes and wall clock) that the recipe cannot change. Audits are performed
by the harness on the trained policy and cannot be influenced by recipe code."""


class RecipeError(Exception):
    """Raised when a recipe fails static checks, loading, or validation."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def static_check(source: str) -> list[str]:
    """Return a list of problems found by AST inspection (empty list = pass)."""
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    problems.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                problems.append(f"forbidden import: from {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                problems.append(f"forbidden call: {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"forbidden dunder attribute access: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__import__"):
            problems.append(f"forbidden name: {node.id}")
    return problems


_LOAD_COUNTER = 0


def load_recipe(path: str | Path):
    """Static-check then import a recipe file as a uniquely named module."""
    global _LOAD_COUNTER
    path = Path(path)
    source = path.read_text()
    problems = static_check(source)
    if problems:
        raise RecipeError(problems)
    _LOAD_COUNTER += 1
    name = f"shinsa_recipe_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RecipeError([f"import failed: {type(e).__name__}: {e}"])
    return module


def _check_schedule_output(sch, gen: int) -> list[str]:
    problems = []
    if not isinstance(sch, dict):
        return [f"schedule({gen}) did not return a dict"]
    ep = sch.get("episodes_per_eval")
    ms = sch.get("max_steps")
    if not isinstance(ep, (int, np.integer)) or not (EPISODES_RANGE[0] <= ep <= EPISODES_RANGE[1]):
        problems.append(f"schedule({gen})['episodes_per_eval']={ep!r} outside {EPISODES_RANGE}")
    if not isinstance(ms, (int, np.integer)) or not (MAX_STEPS_RANGE[0] <= ms <= MAX_STEPS_RANGE[1]):
        problems.append(f"schedule({gen})['max_steps']={ms!r} outside {MAX_STEPS_RANGE}")
    return problems


def validate_recipe(module, obs_dim: int) -> list[str]:
    """Return a list of contract violations (empty list = valid)."""
    problems = []

    opt = getattr(module, "OPTIMIZER", None)
    if not isinstance(opt, dict):
        problems.append("OPTIMIZER missing or not a dict")
    else:
        sigma0 = opt.get("sigma0")
        popsize = opt.get("popsize")
        if not isinstance(sigma0, (int, float, np.floating)) or not (SIGMA0_RANGE[0] <= sigma0 <= SIGMA0_RANGE[1]):
            problems.append(f"OPTIMIZER['sigma0']={sigma0!r} outside {SIGMA0_RANGE}")
        if not isinstance(popsize, (int, np.integer)) or not (POPSIZE_RANGE[0] <= popsize <= POPSIZE_RANGE[1]):
            problems.append(f"OPTIMIZER['popsize']={popsize!r} outside {POPSIZE_RANGE}")

    schedule = getattr(module, "schedule", None)
    if not callable(schedule):
        problems.append("schedule missing or not callable")
    else:
        for gen in (0, 3, 25):
            try:
                problems.extend(_check_schedule_output(schedule(gen), gen))
            except Exception as e:
                problems.append(f"schedule({gen}) raised {type(e).__name__}: {e}")
                break

    shaped = getattr(module, "shaped_reward", None)
    if not callable(shaped):
        problems.append("shaped_reward missing or not callable")
    else:
        obs = np.zeros(obs_dim, dtype=np.float32)
        probes = [
            (obs, 0, 0.0, False, False, 0),
            (obs, 1, 1.0, False, False, 7),
            (obs, 0, -1.0, True, False, 42),
            (obs, 2, 0.5, False, True, 999),
        ]
        for args in probes:
            try:
                v1 = shaped(*args)
                v2 = shaped(*args)
            except Exception as e:
                problems.append(f"shaped_reward raised {type(e).__name__}: {e}")
                break
            if (not isinstance(v1, (int, float, np.floating))
                    or not np.isfinite(v1)
                    or abs(float(v1)) > SHAPED_REWARD_ABS_MAX):
                problems.append(f"shaped_reward returned non-finite/oversized value: {v1!r}")
                break
            if float(v1) != float(v2):
                problems.append("shaped_reward is not deterministic")
                break

    return problems


def clamp_schedule(sch: dict) -> tuple[int, int]:
    """Clamp a schedule dict into hard bounds; returns (episodes_per_eval, max_steps)."""
    ep = int(np.clip(int(sch.get("episodes_per_eval", 4)), *EPISODES_RANGE))
    ms = int(np.clip(int(sch.get("max_steps", 500)), *MAX_STEPS_RANGE))
    return ep, ms
