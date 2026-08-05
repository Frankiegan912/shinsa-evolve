"""Seed recipe for MinAtar Breakout: honest baseline, raw-reward passthrough."""

OPTIMIZER = {"sigma0": 0.5, "popsize": 16}


def schedule(gen):
    return {"episodes_per_eval": 4, "max_steps": 500}


def shaped_reward(obs, action, reward, terminated, truncated, step):
    return reward
