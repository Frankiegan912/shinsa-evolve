"""Paper-quality composite figures (PDF + PNG) for the arXiv draft.

Two-panel layout per figure (Breakout | LunarLander). Colors are the CVD-safe
Okabe-Ito blue/vermillion pair (validated: worst-pair ΔE 21.9 under protanopia);
mode identity is additionally carried by legend + direct labels, never color alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import load_run

C_AUD = "#0072B2"
C_CLA = "#D55E00"
C_BAND = "#bbbbbb"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
})

RUN_LAYOUT = (
    ("MinAtar Breakout", "breakout_audited", "breakout_claimed", "breakout_seed_variance"),
    ("LunarLander-v3", "lunar_audited", "lunar_claimed", "lunar_seed_variance"),
)


def env_paths(runs_root: Path):
    return [
        (title, *(str(runs_root / name) for name in names))
        for title, *names in RUN_LAYOUT
    ]


def _variance_band(run_dir):
    done = load_run(run_dir)["done"]
    a = [r["audited"] for r in done]
    return min(a), max(a)


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)


def fig_gap(runs, out, envs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (title, aud_d, cla_d, _) in zip(axes, envs):
        for d, color, label in ((aud_d, C_AUD, "audited selection"), (cla_d, C_CLA, "claimed selection")):
            gaps = np.array([r["gap"] for r in runs[d]["done"]])
            ax.hist(gaps, bins=16, color=color, alpha=0.55, label=f"{label} ({(gaps > 0).sum()}/{len(gaps)} > 0)")
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("honesty gap (claimed $-$ audited)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("candidates")
    _save(fig, out, "f1_gap")


def fig_scatter(runs, out, envs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    legend_locs = ("upper left", "lower right")
    for ax, loc, (title, aud_d, cla_d, _) in zip(axes, legend_locs, envs):
        for d, color, label in ((aud_d, C_AUD, "audited selection"), (cla_d, C_CLA, "claimed selection")):
            done = runs[d]["done"]
            ax.scatter([r["audited"] for r in done], [r["claimed"] for r in done],
                       s=14, color=color, alpha=0.75, label=label, edgecolors="white", linewidths=0.4)
        lims = ax.get_xlim() + ax.get_ylim()
        lo, hi = min(lims), max(lims)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="#666666")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("audited score")
        ax.legend(fontsize=7, loc=loc)
    # Direct label for the LunarLander poster-child candidate, in the clear space
    # to its upper right (the legend sits lower right).
    lc = runs[envs[1][2]]["done"]
    champ = max(lc, key=lambda r: r["claimed"])
    axes[1].annotate(f"claims {champ['claimed']:+.0f},\nscores {champ['audited']:+.0f}",
                     xy=(champ["audited"], champ["claimed"]), fontsize=7,
                     xytext=(champ["audited"] + 90, champ["claimed"] + 45),
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444"))
    axes[0].set_ylabel("claimed score")
    _save(fig, out, "f2_scatter")


def fig_trajectory(runs, out, envs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (title, aud_d, cla_d, var_d) in zip(axes, envs):
        lo, hi = _variance_band(var_d)
        ax.axhspan(lo, hi, color=C_BAND, alpha=0.35, lw=0, label="seed-recipe retrain range")
        for d, color, label in ((aud_d, C_AUD, "audited selection"), (cla_d, C_CLA, "claimed selection")):
            audited = [r["audited"] for r in runs[d]["done"]]
            best = np.maximum.accumulate(audited)
            ax.plot(range(len(best)), best, color=color, lw=1.6, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("candidate index")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("best audited score so far")
    _save(fig, out, "f3_trajectory")


def fig_outcomes(runs, out, envs):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (title, aud_d, cla_d, _) in zip(axes, envs):
        aud, cla = runs[aud_d]["done"], runs[cla_d]["done"]
        n = min(len(aud), len(cla))
        vals = [
            max(aud[:n], key=lambda r: r["audited"])["audited"],
            max(cla, key=lambda r: r["claimed"])["audited"],
            max(cla, key=lambda r: r["audited"])["audited"],
        ]
        labels = [f"audited pick\n(first {n})", "claimed pick", "claimed run,\nbest available"]
        colors = [C_AUD, C_CLA, C_CLA]
        bars = ax.bar(range(3), vals, 0.55, color=colors)
        bars[2].set_alpha(0.45)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", fontsize=7,
                    va="bottom" if v >= 0 else "top")
        ax.set_xticks(range(3), labels, fontsize=7)
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("audited score of selected candidate")
    _save(fig, out, "f4_outcomes")


def mutation_stats(runs):
    ok0 = total = 0
    secs = []
    for run in runs.values():
        for r in run["done"]:
            m = r.get("mutate")
            if not m or "attempts" not in m:
                continue
            total += 1
            attempts = m["attempts"]
            if attempts and attempts[0].get("ok"):
                ok0 += 1
            secs += [a["seconds"] for a in attempts if "seconds" in a]
    print(f"mutation: {total} mutated candidates, first-attempt acceptance {ok0}/{total} "
          f"({100 * ok0 / max(total, 1):.0f}%), mean CLI latency {np.mean(secs):.0f}s")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("paper/figs"))
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    envs = env_paths(args.runs_root)
    dirs = [d for _, a, c, v in envs for d in (a, c, v)]
    runs = {d: load_run(d) for d in dirs}
    fig_gap(runs, out, envs)
    fig_scatter(runs, out, envs)
    fig_trajectory(runs, out, envs)
    fig_outcomes(runs, out, envs)
    mutation_stats(runs)
    print(f"paper figures written to {out}/")


if __name__ == "__main__":
    main()
