#!/usr/bin/env python3
"""Render one archived LunarLander audit episode without training or model calls.

The output is an illustrative replay, not an additional independent experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from shinsa_evolve.core.policy import DEFAULT_HIDDEN, act_np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/protocol_v3_codex/lunar_replications/audited_seed101"
OUT = ROOT / "assets"
CANDIDATE_ID = 9
AUDIT_EPISODE_INDEX = 0
FRAME_STRIDE = 3


def main() -> None:
    rows = [json.loads(line) for line in (RUN / "archive.jsonl").read_text().splitlines()]
    row = next(record for record in rows if record["id"] == CANDIDATE_ID)
    audit = row["audit"]
    seed = audit["seeds"][AUDIT_EPISODE_INDEX]
    recorded_return = audit["per_episode"][AUDIT_EPISODE_INDEX]
    params = np.load(RUN / f"params/cand_{CANDIDATE_ID:03d}.npy", allow_pickle=False)
    sizes = (8, *DEFAULT_HIDDEN, 4)

    frames: list[Image.Image] = []
    total_reward = 0.0
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=int(seed))
        for step in range(audit["max_steps"]):
            if step % FRAME_STRIDE == 0:
                frame = Image.fromarray(env.render()).convert("RGB")
                frame.thumbnail((400, 400), Image.Resampling.LANCZOS)
                frames.append(frame)
            obs, reward, terminated, truncated, _ = env.step(act_np(params, obs, sizes))
            total_reward += float(reward)
            if terminated or truncated:
                frames.append(Image.fromarray(env.render()).convert("RGB"))
                break
    finally:
        env.close()

    if not np.isclose(total_reward, recorded_return, rtol=0, atol=1e-6):
        raise RuntimeError(f"replay differs from archived audit: {total_reward} != {recorded_return}")
    if not frames:
        raise RuntimeError("renderer returned no frames")

    OUT.mkdir(exist_ok=True)
    still = frames[len(frames) // 5]
    still.save(OUT / "lunarlander_audit_replay.png", optimize=True)
    frames = [frame.resize(still.size, Image.Resampling.LANCZOS) for frame in frames]
    frames[0].save(
        OUT / "lunarlander_audit_replay.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    print(f"Rendered {len(frames)} frames; seed={seed}; raw return={total_reward:.6f}")


if __name__ == "__main__":
    main()
