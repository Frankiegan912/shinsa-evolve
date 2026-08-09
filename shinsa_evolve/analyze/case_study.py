"""Trajectory forensics for one archived LunarLander candidate: how does it earn
its claimed score, and what does it actually do under audit conditions?

Runs the stored best policy on fresh OS-entropy seeds and reports, per episode:
raw return, shaped return under the candidate's own recipe, episode length,
termination vs truncation, leg-contact fraction, altitude/speed while touching,
and engine usage. This is how "suspected reward hack" claims get verified before
they are written down.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.audit import fresh_seeds
from ..core.interface import load_recipe
from ..core.shaping import shaped_return
from ..envs.lunarlander import _run_episode


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cand", type=int, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    params = np.load(run_dir / "params" / f"cand_{args.cand:03d}.npy")
    recipe = load_recipe(run_dir / "recipes" / f"cand_{args.cand:03d}.py")
    sch = recipe.schedule(0)
    print(f"recipe schedule(0)={sch}  OPTIMIZER={recipe.OPTIMIZER}")

    rows = []
    for seed in fresh_seeds(args.episodes):
        obs, act, rew, terminated = _run_episode((params.astype(np.float32), seed, args.max_steps))
        legs = obs[:, 6] + obs[:, 7]
        contact = legs > 0.5
        rows.append({
            "steps": len(rew),
            "terminated": bool(terminated),
            "raw": float(rew.sum()),
            "shaped": shaped_return(recipe, obs, act, rew, terminated),
            "leg_contact_frac": float(contact.mean()),
            "abs_y_while_contact": float(np.abs(obs[contact, 1]).mean()) if contact.any() else None,
            "speed_while_contact": float(np.sqrt(obs[contact, 2] ** 2 + obs[contact, 3] ** 2).mean()) if contact.any() else None,
            "engine_fire_frac": float((act != 0).mean()),
            "main_engine_frac": float((act == 2).mean()),
        })

    for r in rows:
        print(json.dumps(r))
    steps = np.array([r["steps"] for r in rows])
    print("\nAGGREGATE:")
    print(json.dumps({
        "episodes": len(rows),
        "mean_raw": float(np.mean([r["raw"] for r in rows])),
        "mean_shaped": float(np.mean([r["shaped"] for r in rows])),
        "mean_shaping_bonus": float(np.mean([r["shaped"] - r["raw"] for r in rows])),
        "mean_steps": float(steps.mean()),
        "truncated_at_cap": int((~np.array([r["terminated"] for r in rows])).sum()),
        "mean_leg_contact_frac": float(np.mean([r["leg_contact_frac"] for r in rows])),
        "mean_engine_fire_frac": float(np.mean([r["engine_fire_frac"] for r in rows])),
    }, indent=1))


if __name__ == "__main__":
    main()
