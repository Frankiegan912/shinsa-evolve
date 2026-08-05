"""Environment registry. Adapters share one API:

  sizes            fixed MLP layer sizes (obs_dim, *hidden, n_actions)
  obs_dim          flattened observation dimension
  audit_max_steps  fixed step cap used by the audit
  eval_shaped(members, seeds, max_steps, recipe) -> (shaped (P,E), raw (P,E))
  evaluate_raw(params, seeds, max_steps) -> raw (E,)
  close()
"""

from __future__ import annotations

ENV_CONFIGS = {
    "breakout": {
        "max_wall_seconds": 300.0,
        "max_episodes": 8000,
        "audit_episodes": 32,
        "description": (
            "MinAtar Breakout: 10x10x4 binary observation grid, flattened to a 400-dim float "
            "vector; the 4 channels (last axis) are paddle, ball, ball trail, bricks. "
            "3 discrete actions (no-op, left, right). Raw reward: +1 for each brick broken "
            "by the ball. The episode ends when the ball is lost or at the step cap."
        ),
    },
    "lunarlander": {
        "max_wall_seconds": 600.0,
        "max_episodes": 2000,
        "audit_episodes": 32,
        "description": (
            "Gymnasium LunarLander-v3: 8-dim observation (position, velocity, angle, leg "
            "contacts), 4 discrete actions (no-op, left engine, main engine, right engine). "
            "Raw reward: progress toward the pad, +/-100 for landing/crashing, small fuel "
            "penalties; an episode scoring >= 200 is considered solved."
        ),
    },
}


def get_adapter(name: str, pool_size: int | None = None):
    if name == "breakout":
        from .breakout import BreakoutAdapter
        return BreakoutAdapter()
    if name == "lunarlander":
        from .lunarlander import LunarLanderAdapter
        return LunarLanderAdapter(pool_size=pool_size)
    raise ValueError(f"unknown env: {name}")
