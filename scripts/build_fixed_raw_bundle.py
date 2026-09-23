#!/usr/bin/env python3
"""Build a sanitized deterministic bundle for protocol-v4 fixed-raw runs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from build_release_bundle import scan_output, scan_text, sha256, write_deterministic_archive


RUN_DIRS = (
    "pilot_claimed_seed101_retry",
    "pilot_audited_seed101",
    "claimed_seed202",
    "audited_seed202",
    "claimed_seed303",
    "audited_seed303",
)


def source_files(runs_root: Path) -> list[Path]:
    files: list[Path] = []
    for name in RUN_DIRS:
        run = runs_root / name
        required = (
            run / "archive.jsonl",
            run / "run_config.json",
            run / "outer_rng_state.json",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"incomplete source run: {run}")
        files.extend(required)
        files.extend(sorted((run / "recipes").glob("cand_*.py")))
        files.extend(sorted((run / "params").glob("cand_*.npy")))
    return files


def preflight(files: list[Path]) -> None:
    problems = []
    for path in files:
        for problem in scan_text(path, allow_personal_path=path.name == "run_config.json"):
            problems.append(f"{path}: {problem}")
    if problems:
        raise RuntimeError("fixed-raw bundle preflight failed:\n" + "\n".join(problems))


def copy_run(source: Path, target: Path) -> None:
    (target / "recipes").mkdir(parents=True)
    (target / "params").mkdir()
    shutil.copy2(source / "archive.jsonl", target / "archive.jsonl")
    shutil.copy2(source / "outer_rng_state.json", target / "outer_rng_state.json")
    config = json.loads((source / "run_config.json").read_text())
    config["seed_recipe"] = f"recipes/{Path(config['seed_recipe']).name}"
    config["release_sanitized"] = True
    (target / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    for folder, pattern in (("recipes", "cand_*.py"), ("params", "cand_*.npy")):
        for path in sorted((source / folder).glob(pattern)):
            shutil.copy2(path, target / folder / path.name)


def write_manifest(staging: Path, summary: dict) -> None:
    files = [path for path in staging.rglob("*") if path.is_file()]
    manifest = {
        "schema_version": 1,
        "license_included": True,
        "protocol": summary["protocol"],
        "files": {
            path.relative_to(staging).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        },
    }
    (staging / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    runs_root = args.runs_root.resolve()
    summary_path = args.summary.resolve()
    output = args.output.resolve()
    archive = args.archive.resolve() if args.archive else None
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    if archive and archive.suffixes[-2:] != [".tar", ".gz"]:
        raise RuntimeError("--archive must end in .tar.gz")
    if not (repo / "LICENSE").is_file():
        raise RuntimeError("public bundle blocked: project LICENSE is missing")

    summary = json.loads(summary_path.read_text())
    if summary.get("protocol", {}).get("mutation_protocol_version") != 4:
        raise RuntimeError("summary is not for mutation protocol v4")
    if summary.get("scope", {}).get("training_reward") != "fixed_raw":
        raise RuntimeError("summary is not for fixed-raw training")
    summary_run_dirs = {run["run_dir"] for run in summary.get("runs", [])}
    if summary_run_dirs != set(RUN_DIRS):
        raise RuntimeError("summary run set does not match the approved six-run bundle")

    files = source_files(runs_root)
    preflight(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in RUN_DIRS:
            copy_run(runs_root / name, staging / "runs" / name)
        (staging / "evidence").mkdir()
        shutil.copy2(summary_path, staging / "evidence" / summary_path.name)
        shutil.copy2(repo / "LICENSE", staging / "LICENSE")
        (staging / "README.md").write_text(
            "# ShinsaEvolve protocol-v4 fixed-raw replications\n\n"
            "Six independent LunarLander outer-loop archives: three audited-selection and "
            "three claimed-selection runs, each with 15 completed candidates. Training used "
            "raw environment reward rather than candidate reward shaping. Audit seeds, "
            "per-episode returns, recipes, parameters, RNG checkpoints, immutable run "
            "configurations, and model-usage metadata are included. Machine-local seed-recipe "
            "paths were sanitized. The excluded failed first attempt is not part of this "
            "evidence bundle. Inference is descriptive with n=3 outer runs per mode.\n"
        )
        write_manifest(staging, summary)
        scan_output(staging)
        staging.replace(output)
        if archive:
            write_deterministic_archive(output, archive)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"fixed-raw bundle written to {output}")
    if archive:
        print(f"fixed-raw archive written to {archive}")
        print(f"archive sha256: {sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
