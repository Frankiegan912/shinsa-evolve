#!/usr/bin/env python3
"""Render a non-cherry-picked visual comparison of v3 final selections.

For each outer seed, the policy selected by self-reported training score and the
policy selected by independent audit are replayed on the same deterministic display
seed. This is an illustration of archived policies, not an additional outer run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from shinsa_evolve.core.policy import DEFAULT_HIDDEN, act_np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/protocol_v3_codex/lunar_replications"
OUT = ROOT / "assets"
OUTER_SEEDS = (101, 202, 303)
MODES = ("claimed", "audited")
MAX_STEPS = 1000
FRAME_STRIDE = 5
PANEL_SIZE = (320, 240)
HEADER_HEIGHT = 42
ROW_GAP = 10
COL_GAP = 8
HEADER_COLOR = (31, 41, 55)
GUTTER_COLOR = (156, 163, 175)


def display_seed(outer_seed: int) -> int:
    payload = f"shinsa-evolve-v3-display-v1:{outer_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_done(mode: str, outer_seed: int) -> tuple[Path, list[dict]]:
    run = RUNS / f"{mode}_seed{outer_seed}"
    rows = [json.loads(line) for line in (run / "archive.jsonl").read_text().splitlines()]
    done = [row for row in rows if row.get("status") == "done"]
    return run, done


def selected_record(mode: str, outer_seed: int) -> tuple[Path, dict]:
    run, done = load_done(mode, outer_seed)
    return run, max(done, key=lambda row: row[mode])


def replay(run: Path, record: dict, seed: int) -> tuple[list[Image.Image], float]:
    params = np.load(run / f"params/cand_{record['id']:03d}.npy", allow_pickle=False)
    sizes = (8, *DEFAULT_HIDDEN, 4)
    frames: list[Image.Image] = []
    total = 0.0
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(MAX_STEPS):
            if step % FRAME_STRIDE == 0:
                frames.append(Image.fromarray(env.render()).convert("RGB").resize(PANEL_SIZE))
            obs, reward, terminated, truncated, _ = env.step(act_np(params, obs, sizes))
            total += float(reward)
            if terminated or truncated:
                frames.append(Image.fromarray(env.render()).convert("RGB").resize(PANEL_SIZE))
                break
    finally:
        env.close()
    if not frames:
        raise RuntimeError("renderer returned no frames")
    return frames, total


def label_panel(frame: Image.Image, lines: list[str]) -> Image.Image:
    panel = Image.new("RGB", (PANEL_SIZE[0], PANEL_SIZE[1] + HEADER_HEIGHT), HEADER_COLOR)
    panel.paste(frame, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), lines[0], fill="white")
    draw.text((6, 21), lines[1], fill=(229, 231, 235))
    draw.line((0, HEADER_HEIGHT - 1, PANEL_SIZE[0], HEADER_HEIGHT - 1), fill=(96, 165, 250))
    return panel


def plot_incumbent_trajectories() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"claimed": "#d55e00", "audited": "#0072b2"}
    labels = {"claimed": "self-report selection", "audited": "audit selection"}
    x = np.arange(1, 16)
    for mode in MODES:
        trajectories = []
        for outer_seed in OUTER_SEEDS:
            _, done = load_done(mode, outer_seed)
            trajectory = []
            for count in range(1, len(done) + 1):
                incumbent = max(done[:count], key=lambda row: row[mode])
                trajectory.append(incumbent["audited"])
            trajectories.append(trajectory)
            ax.plot(x, trajectory, color=colors[mode], alpha=0.28, linewidth=1.4)
        ax.plot(
            x,
            np.mean(np.asarray(trajectories), axis=0),
            color=colors[mode],
            linewidth=3,
            label=f"{labels[mode]} mean",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Audit quality of the incumbent selected as search progresses")
    ax.set_xlabel("completed candidates in outer run")
    ax.set_ylabel("raw-reward audit of current incumbent")
    ax.set_xticks((1, 3, 5, 7, 9, 11, 13, 15))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "lunarlander_selection_trajectory.png", dpi=220)
    plt.close(fig)


def main() -> None:
    episodes: dict[tuple[int, str], dict] = {}
    for outer_seed in OUTER_SEEDS:
        seed = display_seed(outer_seed)
        for mode in MODES:
            run, record = selected_record(mode, outer_seed)
            frames, total = replay(run, record, seed)
            episodes[(outer_seed, mode)] = {
                "frames": frames,
                "return": total,
                "record": record,
                "display_seed": seed,
            }

    frame_count = max(len(episode["frames"]) for episode in episodes.values())
    combined: list[Image.Image] = []
    for index in range(frame_count):
        panel_height = PANEL_SIZE[1] + HEADER_HEIGHT
        canvas = Image.new(
            "RGB",
            (
                PANEL_SIZE[0] * len(MODES) + COL_GAP * (len(MODES) - 1),
                panel_height * len(OUTER_SEEDS) + ROW_GAP * (len(OUTER_SEEDS) - 1),
            ),
            GUTTER_COLOR,
        )
        for row_index, outer_seed in enumerate(OUTER_SEEDS):
            for col_index, mode in enumerate(MODES):
                episode = episodes[(outer_seed, mode)]
                frames = episode["frames"]
                frame = frames[min(index, len(frames) - 1)]
                record = episode["record"]
                selection = "self-report" if mode == "claimed" else "audit gate"
                lines = [
                    f"outer {outer_seed} | {selection} | reported {record['claimed']:.3g}",
                    f"display {episode['return']:.1f} | 32-ep audit {record['audited']:.1f}",
                ]
                panel = label_panel(frame, lines)
                canvas.paste(
                    panel,
                    (
                        col_index * (PANEL_SIZE[0] + COL_GAP),
                        row_index * (panel_height + ROW_GAP),
                    ),
                )
        combined.append(canvas)

    OUT.mkdir(exist_ok=True)
    still = combined[min(len(combined) // 5, len(combined) - 1)]
    still.save(OUT / "lunarlander_selection_comparison.png", optimize=True)
    combined[0].save(
        OUT / "lunarlander_selection_comparison.gif",
        save_all=True,
        append_images=combined[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    plot_incumbent_trajectories()
    for outer_seed in OUTER_SEEDS:
        left = episodes[(outer_seed, "claimed")]
        right = episodes[(outer_seed, "audited")]
        print(
            f"outer={outer_seed} display_seed={left['display_seed']} "
            f"self_report_return={left['return']:.6f} audit_return={right['return']:.6f}"
        )


if __name__ == "__main__":
    main()
