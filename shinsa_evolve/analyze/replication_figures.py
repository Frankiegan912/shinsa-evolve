"""Figures for the protocol-v3 study and protocol-v4 fixed-raw ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C_AUD = "#0072B2"
C_CLA = "#D55E00"
C_ORACLE = "#009E73"
C_TIE = "#CC79A7"
C_GRAY = "#777777"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def load_summary(path: Path) -> dict:
    summary = json.loads(path.read_text())
    if summary.get("protocol", {}).get("mutation_protocol_version") != 3:
        raise RuntimeError("expected protocol-v3 summary")
    return summary


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=240)
    plt.close(fig)


def selected_outcomes(summary: dict, out: Path) -> None:
    runs = summary["runs"]
    seeds = summary["scope"]["rng_seeds"]
    by_key = {(run["mode"], run["rng_seed"]): run for run in runs}
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    x = np.arange(len(seeds))
    audited = [by_key[("audited", seed)]["selected"]["audited"] for seed in seeds]
    claimed = [by_key[("claimed", seed)]["selected"]["audited"] for seed in seeds]
    for i in range(len(seeds)):
        ax.plot([x[i] - 0.13, x[i] + 0.13], [audited[i], claimed[i]],
                color="#bbbbbb", lw=1.0, zorder=1)
    ax.scatter(x - 0.13, audited, s=45, color=C_AUD, marker="o",
               label="audited selection", zorder=3)
    ax.scatter(x + 0.13, claimed, s=50, color=C_CLA, marker="s",
               label="claimed selection", zorder=3)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("outer-loop RNG seed")
    ax.set_ylabel("raw-reward audit of final pick")
    ax.legend(loc="lower left")
    ax.set_title("Final selected policy in each independent outer run")
    save(fig, out, "v3_selected_outcomes")


def claimed_regret(summary: dict, out: Path) -> None:
    runs = [run for run in summary["runs"] if run["mode"] == "claimed"]
    runs.sort(key=lambda run: run["rng_seed"])
    x = np.arange(len(runs))
    picked = [run["selected"]["audited"] for run in runs]
    tie_best = [run["tie_robust"]["best_audited"] for run in runs]
    oracle = [run["oracle_by_audit"]["audited"] for run in runs]
    fig, ax = plt.subplots(figsize=(5.7, 3.0))
    for i, run in enumerate(runs):
        ax.plot([x[i], x[i]], [picked[i], oracle[i]], color=C_GRAY, lw=1.2, zorder=1)
        ax.text(x[i] + 0.06, (picked[i] + oracle[i]) / 2,
                f"actual {run['selection_regret']:.0f}\n"
                f"tie-best {run['tie_robust']['minimum_selection_regret']:.0f}",
                fontsize=7.5, va="center")
    ax.scatter(x, oracle, s=48, color=C_ORACLE, marker="o",
               label="best audited candidate available", zorder=3)
    ax.scatter(x, picked, s=50, color=C_CLA, marker="s",
               label="candidate selected by claimed score", zorder=3)
    ax.scatter(x, tie_best, s=46, color=C_TIE, marker="D",
               label="best audit among top-claimed ties", zorder=4)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(x, [str(run["rng_seed"]) for run in runs])
    ax.set_xlabel("outer-loop RNG seed")
    ax.set_ylabel("raw-reward audit")
    ax.set_title("Claim saturation makes the final ranking tie-sensitive")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, out, "v3_claimed_regret")


def candidate_scale(runs_root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 3.2))
    for mode, color, marker in (("audited", C_AUD, "o"), ("claimed", C_CLA, "s")):
        rows = []
        for seed in (101, 202, 303):
            archive = runs_root / f"{mode}_seed{seed}" / "archive.jsonl"
            rows.extend(
                row for row in map(json.loads, archive.read_text().splitlines())
                if row.get("status") == "done"
            )
        ax.scatter([row["audited"] for row in rows], [row["claimed"] for row in rows],
                   s=22, color=color, marker=marker, alpha=0.72,
                   edgecolors="white", linewidths=0.35, label=f"{mode} selection")
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.axvline(0, color="#444444", lw=0.7)
    ax.set_xlabel("raw-reward audit")
    ax.set_ylabel("candidate-reported training score (symlog)")
    ax.set_title("Claimed selection drives self-reports to the reward-scale ceiling")
    ax.legend(loc="lower right")
    save(fig, out, "v3_candidate_scale")


def mechanism_summary(summary: dict, runs_root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    for mode, color, marker in (("audited", C_AUD, "o"), ("claimed", C_CLA, "s")):
        rows = []
        for seed in (101, 202, 303):
            archive = runs_root / f"{mode}_seed{seed}" / "archive.jsonl"
            rows.extend(
                row for row in map(json.loads, archive.read_text().splitlines())
                if row.get("status") == "done"
            )
        ax.scatter([row["audited"] for row in rows], [row["claimed"] for row in rows],
                   s=15, color=color, marker=marker, alpha=0.72,
                   edgecolors="white", linewidths=0.25, label=mode)
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.axvline(0, color="#444444", lw=0.6)
    ax.set_xlabel("raw-reward audit")
    ax.set_ylabel("self-report (symlog)")
    ax.set_title("(a) Candidate score scale", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    runs = sorted(
        (run for run in summary["runs"] if run["mode"] == "claimed"),
        key=lambda run: run["rng_seed"],
    )
    x = np.arange(len(runs))
    picked = [run["selected"]["audited"] for run in runs]
    tie_best = [run["tie_robust"]["best_audited"] for run in runs]
    oracle = [run["oracle_by_audit"]["audited"] for run in runs]
    for i, run in enumerate(runs):
        ax.plot([x[i], x[i]], [picked[i], oracle[i]], color=C_GRAY, lw=1.0, zorder=1)
        ax.text(x[i] + 0.05, (picked[i] + oracle[i]) / 2,
                f"{run['selection_regret']:.0f} / "
                f"{run['tie_robust']['minimum_selection_regret']:.0f}",
                fontsize=6.5, va="center")
    ax.scatter(x, oracle, s=30, color=C_ORACLE, marker="o", label="archive oracle", zorder=3)
    ax.scatter(x, picked, s=32, color=C_CLA, marker="s", label="actual pick", zorder=3)
    ax.scatter(x, tie_best, s=30, color=C_TIE, marker="D", label="best top-score tie", zorder=4)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(x, [str(run["rng_seed"]) for run in runs])
    ax.set_xlabel("outer-loop RNG seed")
    ax.set_ylabel("raw-reward audit")
    ax.set_title("(b) Actual / tie-best regret", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    save(fig, out, "v3_mechanism")


def reward_source_ablation(v3: dict, v4: dict, out: Path) -> None:
    """Compare final picks and claimed-mode regret across reward-source protocols."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    protocols = (("candidate shaped", v3), ("fixed raw", v4))
    offsets = {"audited": -0.13, "claimed": 0.13}
    styles = {
        "audited": (C_AUD, "o", "audited selection"),
        "claimed": (C_CLA, "s", "claimed selection"),
    }

    ax = axes[0]
    for pidx, (_, summary) in enumerate(protocols):
        by_key = {(run["mode"], run["rng_seed"]): run for run in summary["runs"]}
        for mode in ("audited", "claimed"):
            color, marker, label = styles[mode]
            values = [by_key[(mode, seed)]["selected"]["audited"]
                      for seed in (101, 202, 303)]
            xs = np.full(3, pidx + offsets[mode])
            ax.scatter(xs, values, s=34, color=color, marker=marker,
                       edgecolors="white", linewidths=0.35,
                       label=label if pidx == 0 else None, zorder=3)
            ax.plot([pidx + offsets[mode]] * 2, [min(values), max(values)],
                    color=color, alpha=0.35, lw=1.0, zorder=1)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks([0, 1], ["candidate\nshaped", "fixed\nraw"])
    ax.set_ylabel("raw-reward audit of final pick")
    ax.set_title("(a) Final selected policies", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    for pidx, (name, summary) in enumerate(protocols):
        runs = sorted((run for run in summary["runs"] if run["mode"] == "claimed"),
                      key=lambda run: run["rng_seed"])
        regrets = [run["selection_regret"] for run in runs]
        ax.scatter(np.full(3, pidx), regrets, s=36, color=C_CLA, marker="s",
                   edgecolors="white", linewidths=0.35, zorder=3)
        mean = float(np.mean(regrets))
        ax.plot([pidx - 0.18, pidx + 0.18], [mean, mean], color="#222222", lw=1.5)
        ax.text(pidx + 0.06, mean, f"mean {mean:.0f}", fontsize=7, va="bottom")
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks([0, 1], ["candidate\nshaped", "fixed\nraw"])
    ax.set_ylabel("claimed-run selection regret")
    ax.set_title("(b) Mis-ranking remains after ablation", fontsize=9)
    save(fig, out, "v3_v4_ablation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--fixed-raw-summary", type=Path)
    parser.add_argument("--out", type=Path, default=Path("paper/figs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    selected_outcomes(summary, args.out)
    claimed_regret(summary, args.out)
    candidate_scale(args.runs_root, args.out)
    mechanism_summary(summary, args.runs_root, args.out)
    if args.fixed_raw_summary:
        fixed_raw = json.loads(args.fixed_raw_summary.read_text())
        if fixed_raw.get("protocol", {}).get("mutation_protocol_version") != 4:
            raise RuntimeError("expected protocol-v4 fixed-raw summary")
        reward_source_ablation(summary, fixed_raw, args.out)
    print(f"replication figures written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
