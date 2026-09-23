"""Inner training loop: fixed small MLP optimized with separable CMA-ES.

Separable (diagonal) CMA-ES is used because the flat parameter vector is in the
10^3..10^4 range, where full-covariance CMA-ES is intractable. The recipe controls
only OPTIMIZER / schedule() / shaped_reward(); episode and wall-clock budgets are
hard caps enforced here.

The *claimed* score of a candidate is the best training fitness as measured by the
recipe's own signal (its shaped reward, its episode counts, the fixed per-candidate
training seed pool). It is deliberately candidate-influenced; the audit is not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from cmaes import SepCMA

from .interface import SIGMA0_RANGE, POPSIZE_RANGE, clamp_schedule
from .policy import param_count


@dataclass
class TrainBudget:
    max_episodes: int
    max_wall_seconds: float


@dataclass
class TrainResult:
    best_params: np.ndarray
    claimed_score: float
    train_raw_of_best: float
    best_generation: int
    generations: int
    episodes_used: int
    wall_seconds: float
    train_seed: int
    seed_pool: list[int]
    history: list = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "claimed_score": self.claimed_score,
            "train_raw_of_best": self.train_raw_of_best,
            "best_generation": self.best_generation,
            "generations": self.generations,
            "episodes_used": self.episodes_used,
            "wall_seconds": round(self.wall_seconds, 2),
            "train_seed": self.train_seed,
            "seed_pool": self.seed_pool,
            "history": self.history,
        }


def train(recipe, adapter, budget: TrainBudget, train_seed: int) -> TrainResult:
    sizes = adapter.sizes
    dim = param_count(sizes)

    sigma0 = float(np.clip(float(recipe.OPTIMIZER["sigma0"]), *SIGMA0_RANGE))
    popsize = int(np.clip(int(recipe.OPTIMIZER["popsize"]), *POPSIZE_RANGE))
    es = SepCMA(mean=np.zeros(dim), sigma=sigma0, population_size=popsize, seed=train_seed)

    # Fixed per-candidate seed pool, reused every generation. Reuse is deliberate:
    # it leaves room for seed overfitting, one honest source of the claimed/audited gap.
    seed_pool = [(train_seed * 100003 + 7919 * e) % (2**31 - 1) for e in range(16)]

    best_fitness = -np.inf
    best_params = np.zeros(dim)
    best_raw = -np.inf
    best_generation = -1
    history = []
    episodes_used = 0
    gen = 0
    t0 = time.monotonic()

    while True:
        try:
            episodes, max_steps = clamp_schedule(recipe.schedule(gen))
        except Exception:
            episodes, max_steps = 4, 500
        # Always allow generation 0 so every candidate produces a policy.
        if gen > 0 and episodes_used + popsize * episodes > budget.max_episodes:
            break

        members = [es.ask() for _ in range(popsize)]
        seeds = seed_pool[:episodes]
        shaped, raw = adapter.eval_shaped(members, seeds, max_steps, recipe)  # (P, E) each
        fitness = np.nan_to_num(shaped.mean(axis=1), nan=-1e9, posinf=-1e9, neginf=-1e9)
        es.tell([(members[i], -float(fitness[i])) for i in range(popsize)])

        gi = int(np.argmax(fitness))
        if fitness[gi] > best_fitness:
            best_fitness = float(fitness[gi])
            best_params = np.array(members[gi])
            best_raw = float(raw[gi].mean())
            best_generation = gen

        episodes_used += popsize * episodes
        history.append({
            "gen": gen,
            "best_fitness": float(fitness[gi]),
            "mean_fitness": float(fitness.mean()),
            "episodes_per_eval": episodes,
            "max_steps": max_steps,
        })
        gen += 1
        if episodes_used >= budget.max_episodes:
            break
        if time.monotonic() - t0 >= budget.max_wall_seconds:
            break

    return TrainResult(
        best_params=best_params,
        claimed_score=float(best_fitness),
        train_raw_of_best=float(best_raw),
        best_generation=best_generation,
        generations=gen,
        episodes_used=episodes_used,
        wall_seconds=time.monotonic() - t0,
        train_seed=int(train_seed),
        seed_pool=[int(seed) for seed in seed_pool],
        history=history,
    )
