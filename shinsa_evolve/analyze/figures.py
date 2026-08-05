"""Main figures from one or more run directories (basic versions; polish in the
analysis phase).

  F1 selection trajectory: best-audited-so-far vs candidate index, per mode
  F2 honesty scatter: claimed vs audited, all candidates
  F3 honesty gap distribution
  F4 selection failure: audited score of the pick-by-claimed vs the best available,
     plus a budget-aligned (truncated-to-N) mode comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    records = []
    with open(run_dir / "archive.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    done = sorted((r for r in records if r.get("status") == "done"), key=lambda r: r["id"])
    cfg = json.loads((run_dir / "run_config.json").read_text())
    return {"dir": run_dir, "cfg": cfg, "done": done,
            "label": f"{cfg['env']}/{cfg['mode']}"}


def _best_so_far(values):
    out, best = [], -np.inf
    for v in values:
        best = max(best, v)
        out.append(best)
    return out


def fig_selection_trajectory(runs, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for run in runs:
        audited = [r["audited"] for r in run["done"]]
        ax.plot(range(len(audited)), _best_so_far(audited), label=run["label"])
    ax.set_xlabel("candidate index")
    ax.set_ylabel("best audited score so far")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "f1_selection_trajectory.png", dpi=200)


def fig_honesty_scatter(runs, out: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for run in runs:
        c = [r["claimed"] for r in run["done"]]
        a = [r["audited"] for r in run["done"]]
        ax.scatter(a, c, s=14, alpha=0.7, label=run["label"])
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="claimed = audited")
    ax.set_xlabel("audited score (raw, fresh seeds)")
    ax.set_ylabel("claimed score (recipe-reported)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "f2_honesty_scatter.png", dpi=200)


def fig_gap_distribution(runs, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for run in runs:
        gaps = [r["gap"] for r in run["done"]]
        ax.hist(gaps, bins=20, alpha=0.5, label=run["label"])
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("honesty gap (claimed - audited)")
    ax.set_ylabel("candidates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "f3_gap_distribution.png", dpi=200)


def fig_selection_failure(runs, out: Path):
    """For claimed-mode runs: what the claimed ranking picked vs what was available;
    plus a budget-aligned comparison across modes (truncate all runs to min N)."""
    claimed_runs = [r for r in runs if r["cfg"]["mode"] == "claimed"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    labels, picked, best_avail = [], [], []
    for run in claimed_runs:
        done = run["done"]
        if not done:
            continue
        pick = max(done, key=lambda r: r["claimed"])
        avail = max(done, key=lambda r: r["audited"])
        labels.append(run["label"])
        picked.append(pick["audited"])
        best_avail.append(avail["audited"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, picked, 0.4, label="picked by claimed")
    ax.bar(x + 0.2, best_avail, 0.4, label="best available (audited)")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("audited score")
    ax.set_title("claimed-mode selection failure")
    ax.legend(fontsize=8)

    ax = axes[1]
    by_env = {}
    for run in runs:
        by_env.setdefault(run["cfg"]["env"], []).append(run)
    labels, vals = [], []
    for env, env_runs in by_env.items():
        n = min(len(r["done"]) for r in env_runs)
        for run in env_runs:
            done = run["done"][:n]
            mode = run["cfg"]["mode"]
            key = "claimed" if mode == "claimed" else "audited"
            pick = max(done, key=lambda r: r[key])
            labels.append(f"{env}/{mode}\n(n={n})")
            vals.append(pick["audited"])
    ax.bar(range(len(labels)), vals)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylabel("audited score of final pick")
    ax.set_title("budget-aligned comparison")

    fig.tight_layout()
    fig.savefig(out / "f4_selection_failure.png", dpi=200)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="analysis_out")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(d) for d in args.runs]
    fig_selection_trajectory(runs, out)
    fig_honesty_scatter(runs, out)
    fig_gap_distribution(runs, out)
    fig_selection_failure(runs, out)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
