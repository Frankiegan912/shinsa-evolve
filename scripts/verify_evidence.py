#!/usr/bin/env python3
"""Read-only verifier for the archived evidence behind the draft paper.

The verifier never imports or executes generated recipe code.  It parses the
small schedule() subset needed for the paper's schedule(0) table with a limited
AST evaluator, recomputes the headline statistics, and hashes the evidence
trees while excluding bytecode caches.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import statistics
import subprocess
from pathlib import Path
from typing import Any


MAIN_RUNS = (
    "breakout_audited",
    "breakout_claimed",
    "lunar_audited",
    "lunar_claimed",
)
VARIANCE_RUNS = ("breakout_seed_variance", "lunar_seed_variance")

EXPECTED = {
    "breakout_audited": {
        "done": 60,
        "gap_mean": 0.37285626679020367,
        "gap_median": 0.09825411502682635,
        "gap_positive": 36,
        "gap_max": 3.6064845736463047,
        "pick_by_mode_audited": 4.9375,
    },
    "breakout_claimed": {
        "done": 20,
        "gap_mean": 1.1029486985125987,
        "gap_median": 0.8357753086419741,
        "gap_positive": 18,
        "gap_max": 2.7745062366411615,
        "pick_by_mode_audited": 4.3125,
    },
    "lunar_audited": {
        "done": 40,
        "gap_mean": 140.78929730214293,
        "gap_median": 129.02329119387224,
        "gap_positive": 38,
        "gap_max": 523.9911793306073,
        "pick_by_mode_audited": -45.13163044885479,
    },
    "lunar_claimed": {
        "done": 15,
        "gap_mean": 241.274205546156,
        "gap_median": 222.92285639837027,
        "gap_positive": 15,
        "gap_max": 552.7856795887571,
        "pick_by_mode_audited": -118.81421341885977,
    },
}


def read_archive(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            invalid += 1
    return records, invalid


def tree_digest(run_dir: Path) -> dict[str, Any]:
    files = [run_dir / "archive.jsonl", run_dir / "run_config.json"]
    files += sorted((run_dir / "recipes").glob("cand_*.py"))
    files += sorted((run_dir / "params").glob("cand_*.npy"))
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(files, key=lambda p: p.relative_to(run_dir).as_posix()):
        rel = path.relative_to(run_dir).as_posix().encode()
        data = path.read_bytes()
        digest.update(rel + b"\0" + str(len(data)).encode() + b"\0" + data)
        total_bytes += len(data)
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
COMPARE_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
SAFE_CALLS = {"min": min, "max": max, "int": int, "float": float}


def eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Dict):
        return {
            eval_expr(k, env): eval_expr(v, env)
            for k, v in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [eval_expr(x, env) for x in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        return BIN_OPS[type(node.op)](eval_expr(node.left, env), eval_expr(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_expr(node.operand, env)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        op = COMPARE_OPS.get(type(node.ops[0]))
        if op is not None:
            return op(eval_expr(node.left, env), eval_expr(node.comparators[0], env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = SAFE_CALLS.get(node.func.id)
        if fn is not None and not node.keywords:
            return fn(*(eval_expr(arg, env) for arg in node.args))
    raise ValueError(f"unsupported schedule expression: {ast.dump(node, include_attributes=False)}")


def eval_statements(statements: list[ast.stmt], env: dict[str, Any]) -> Any:
    for statement in statements:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = eval_expr(statement.value, env)
            if isinstance(target, ast.Name):
                env[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and all(
                isinstance(item, ast.Name) for item in target.elts
            ):
                if len(target.elts) != len(value):
                    raise ValueError("schedule unpacking length mismatch")
                for item, part in zip(target.elts, value, strict=True):
                    env[item.id] = part
            else:
                raise ValueError("unsupported schedule assignment target")
        elif isinstance(statement, ast.If):
            branch = statement.body if eval_expr(statement.test, env) else statement.orelse
            result = eval_statements(branch, env)
            if result is not None:
                return result
        elif isinstance(statement, ast.Return):
            return eval_expr(statement.value, env)
        else:
            raise ValueError(f"unsupported schedule statement: {type(statement).__name__}")
    return None


def schedule_zero(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    fn = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "schedule"),
        None,
    )
    if fn is None:
        raise ValueError(f"schedule() missing: {path}")
    result = eval_statements(fn.body, {"gen": 0})
    if not isinstance(result, dict):
        raise ValueError(f"schedule(0) did not produce a dict: {path}")
    return {
        "episodes_per_eval": int(result["episodes_per_eval"]),
        "max_steps": int(result["max_steps"]),
    }


def summarize_main_run(run_dir: Path) -> dict[str, Any]:
    records, invalid = read_archive(run_dir / "archive.jsonl")
    done = [r for r in records if r.get("status") == "done"]
    failed = [r for r in records if r.get("status") != "done"]
    gaps = [float(r["gap"]) for r in done]
    mode = json.loads((run_dir / "run_config.json").read_text())["mode"]
    selection_key = "claimed" if mode == "claimed" else "audited"
    pick = max(done, key=lambda r: r[selection_key])
    best_audited = max(done, key=lambda r: r["audited"])
    recipes = sorted(
        (run_dir / "recipes").glob("cand_*.py"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    params = sorted((run_dir / "params").glob("cand_*.npy"))
    missing_recipes = [r["id"] for r in done if not (run_dir / r["recipe"]).is_file()]
    missing_params = [
        r["id"] for r in done
        if not (run_dir / "params" / f"cand_{r['id']:03d}.npy").is_file()
    ]
    schedules = [schedule_zero(path) for path in recipes]
    last = schedules[-10:]
    mutated = [r for r in done if (r.get("mutate") or {}).get("attempts")]
    attempts = [attempt for r in mutated for attempt in r["mutate"]["attempts"]]
    return {
        "records": len(records),
        "done": len(done),
        "failed": len(failed),
        "invalid_json_lines": invalid,
        "completeness": {
            "recipe_files": len(recipes),
            "parameter_files": len(params),
            "missing_recipe_ids": missing_recipes,
            "missing_parameter_ids": missing_params,
        },
        "gap_mean": statistics.fmean(gaps),
        "gap_median": statistics.median(gaps),
        "gap_positive": sum(gap > 0 for gap in gaps),
        "gap_max": max(gaps),
        "pick_by_mode": {
            "id": pick["id"],
            "claimed": pick["claimed"],
            "audited": pick["audited"],
        },
        "best_available_by_audit": {
            "id": best_audited["id"],
            "audited": best_audited["audited"],
        },
        "schedule_zero": {
            "episodes_all_mean": statistics.fmean(s["episodes_per_eval"] for s in schedules),
            "episodes_last10_mean": statistics.fmean(s["episodes_per_eval"] for s in last),
            "max_steps_all_mean": statistics.fmean(s["max_steps"] for s in schedules),
            "max_steps_last10_mean": statistics.fmean(s["max_steps"] for s in last),
        },
        "mutation": {
            "completed_mutations": len(mutated),
            "first_attempt_ok": sum(r["mutate"]["attempts"][0].get("ok") is True for r in mutated),
            "recorded_attempts": len(attempts),
            "mean_cli_seconds": statistics.fmean(float(a["seconds"]) for a in attempts),
        },
        "evidence_tree": tree_digest(run_dir),
    }


def summarize_variance_run(run_dir: Path) -> dict[str, Any]:
    records, invalid = read_archive(run_dir / "archive.jsonl")
    done = [r for r in records if r.get("status") == "done"]
    scores = [float(r["audited"]) for r in done]
    recipes = sorted((run_dir / "recipes").glob("cand_*.py"))
    params = sorted((run_dir / "params").glob("cand_*.npy"))
    return {
        "done": len(done),
        "invalid_json_lines": invalid,
        "completeness": {
            "recipe_files": len(recipes),
            "parameter_files": len(params),
        },
        "audited_scores": scores,
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.fmean(scores),
        "population_std": statistics.pstdev(scores),
        "evidence_tree": tree_digest(run_dir),
    }


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def verify_expected(main: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, expected in EXPECTED.items():
        actual = main[name]
        for key in ("done", "gap_positive"):
            if actual[key] != expected[key]:
                failures.append(f"{name}.{key}: {actual[key]} != {expected[key]}")
        for key in ("gap_mean", "gap_median", "gap_max"):
            if not close(float(actual[key]), float(expected[key])):
                failures.append(f"{name}.{key}: {actual[key]} != {expected[key]}")
        if not close(
            float(actual["pick_by_mode"]["audited"]),
            float(expected["pick_by_mode_audited"]),
        ):
            failures.append(f"{name}.pick_by_mode.audited differs from expected")
        completeness = actual["completeness"]
        if completeness["recipe_files"] != actual["done"]:
            failures.append(f"{name}: recipe count does not match completed candidates")
        if completeness["parameter_files"] != actual["done"]:
            failures.append(f"{name}: parameter count does not match completed candidates")
        if completeness["missing_recipe_ids"] or completeness["missing_parameter_ids"]:
            failures.append(f"{name}: completed records reference missing artifacts")
        if actual["invalid_json_lines"]:
            failures.append(f"{name}: archive contains invalid JSON lines")
    return failures


def build_manifest(repo: Path, runs_root: Path | None = None) -> dict[str, Any]:
    runs = runs_root or repo / "runs"
    main = {name: summarize_main_run(runs / name) for name in MAIN_RUNS}
    variance = {
        name: summarize_variance_run(runs / name)
        for name in VARIANCE_RUNS
    }
    matched = {
        "breakout": {
            "n": 20,
            "audited_mode_pick": max(
                [
                    r
                    for r in read_archive(runs / "breakout_audited/archive.jsonl")[0]
                    if r.get("status") == "done"
                ][:20],
                key=lambda r: r["audited"],
            )["audited"],
            "claimed_mode_pick": main["breakout_claimed"]["pick_by_mode"]["audited"],
        },
        "lunarlander": {
            "n": 15,
            "audited_mode_pick": max(
                [
                    r
                    for r in read_archive(runs / "lunar_audited/archive.jsonl")[0]
                    if r.get("status") == "done"
                ][:15],
                key=lambda r: r["audited"],
            )["audited"],
            "claimed_mode_pick": main["lunar_claimed"]["pick_by_mode"]["audited"],
        },
    }
    mutation_attempts = [
        run["mutation"] for run in main.values()
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_revision": commit,
        "scope": {
            "main_runs": list(MAIN_RUNS),
            "variance_runs": list(VARIANCE_RUNS),
            "excluded": ["smoke runs", "bytecode caches", "virtual environment"],
        },
        "main_runs": main,
        "variance_runs": variance,
        "matched_budget": matched,
        "mutation_total": {
            "completed_mutations": sum(x["completed_mutations"] for x in mutation_attempts),
            "first_attempt_ok": sum(x["first_attempt_ok"] for x in mutation_attempts),
            "recorded_attempts": sum(x["recorded_attempts"] for x in mutation_attempts),
            "mean_cli_seconds": statistics.fmean(
                float(a["seconds"])
                for name in MAIN_RUNS
                for r in read_archive(runs / name / "archive.jsonl")[0]
                if r.get("status") == "done" and (r.get("mutate") or {}).get("attempts")
                for a in r["mutate"]["attempts"]
            ),
        },
    }
    failures = verify_expected(main)
    for name, summary in variance.items():
        if summary["done"] != 4:
            failures.append(f"{name}: expected four completed seed trainings")
        if summary["completeness"] != {"recipe_files": 1, "parameter_files": 4}:
            failures.append(f"{name}: incomplete seed-variance artifacts")
        if summary["invalid_json_lines"]:
            failures.append(f"{name}: archive contains invalid JSON lines")
    manifest["verification"] = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="optional released runs directory instead of REPO_ROOT/runs",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    runs_root = args.runs_root.resolve() if args.runs_root else None
    manifest = build_manifest(args.repo_root.resolve(), runs_root)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if manifest["verification"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
