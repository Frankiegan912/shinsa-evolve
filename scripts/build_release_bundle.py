#!/usr/bin/env python3
"""Build a sanitized, checksummed data bundle from the recovered local runs.

Public bundle creation is blocked until a project LICENSE exists.  The
--allow-unlicensed-preview flag is intentionally limited to local review.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path


RUNS = (
    "breakout_audited",
    "breakout_claimed",
    "lunar_audited",
    "lunar_claimed",
    "breakout_seed_variance",
    "lunar_seed_variance",
)
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".txt"}
DEFAULT_DENY_PATTERN = r"confidential|proprietary|do.?not.?distribute"
RESTRICTED_RE = re.compile(
    os.environ.get("SHINSA_RELEASE_DENY_PATTERN", DEFAULT_DENY_PATTERN),
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|API[_-]?KEY|TOKEN)\s*[=:]",
    re.IGNORECASE,
)
PERSONAL_PATH_RE = re.compile(r"/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for name in RUNS:
        run = repo / "runs" / name
        required = (run / "archive.jsonl", run / "run_config.json")
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"incomplete source run: {run}")
        files.extend(required)
        files.extend(sorted((run / "recipes").glob("cand_*.py")))
        files.extend(sorted((run / "params").glob("cand_*.npy")))
    files.extend(sorted((repo / "recipes").glob("seed_*.py")))
    files.extend((repo / "evidence" / "evidence_manifest.json",
                  repo / "evidence" / "CLAIM_AUDIT.md"))
    return files


def scan_text(path: Path, *, allow_personal_path: bool = False) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    text = path.read_text(errors="replace")
    problems = []
    if RESTRICTED_RE.search(text):
        problems.append("restricted-material keyword")
    if SECRET_RE.search(text):
        problems.append("secret-like value")
    if not allow_personal_path and PERSONAL_PATH_RE.search(text):
        problems.append("personal absolute path")
    return problems


def preflight(repo: Path, files: list[Path]) -> None:
    problems = []
    for path in files:
        # Source run_config paths are expected and are sanitized during copying.
        allow_path = path.name == "run_config.json"
        for problem in scan_text(path, allow_personal_path=allow_path):
            problems.append(f"{path.relative_to(repo)}: {problem}")
    if problems:
        raise RuntimeError("release preflight failed:\n" + "\n".join(problems))


def copy_sanitized(repo: Path, staging: Path) -> None:
    for name in RUNS:
        source = repo / "runs" / name
        target = staging / "runs" / name
        (target / "recipes").mkdir(parents=True)
        (target / "params").mkdir()
        shutil.copy2(source / "archive.jsonl", target / "archive.jsonl")
        config = json.loads((source / "run_config.json").read_text())
        seed_name = Path(config["seed_recipe"]).name
        config["seed_recipe"] = f"recipes/{seed_name}"
        config["release_sanitized"] = True
        (target / "run_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        for path in sorted((source / "recipes").glob("cand_*.py")):
            shutil.copy2(path, target / "recipes" / path.name)
        for path in sorted((source / "params").glob("cand_*.npy")):
            shutil.copy2(path, target / "params" / path.name)

    (staging / "recipes").mkdir()
    for path in sorted((repo / "recipes").glob("seed_*.py")):
        shutil.copy2(path, staging / "recipes" / path.name)
    (staging / "evidence").mkdir()
    for name in ("evidence_manifest.json", "CLAIM_AUDIT.md"):
        shutil.copy2(repo / "evidence" / name, staging / "evidence" / name)


def write_bundle_metadata(repo: Path, staging: Path, licensed: bool) -> None:
    readme = """# ShinsaEvolve recovered experiment data

This bundle contains the six historical runs used by the August 2026 draft:
four selection runs and two seed-recipe variance runs. The historical archives
predate archive schema v2, so audit seeds, per-episode returns, and full training
history are not present. See `evidence/CLAIM_AUDIT.md` for supported claim bounds.

Machine-local paths were replaced in copied run configurations. Runtime logs,
caches, smoke runs, and local editor state are excluded.
"""
    (staging / "README.md").write_text(readme)
    if licensed:
        shutil.copy2(repo / "LICENSE", staging / "LICENSE")

    files = [p for p in staging.rglob("*") if p.is_file()]
    manifest = {
        "schema_version": 1,
        "source_revision": json.loads(
            (repo / "evidence" / "evidence_manifest.json").read_text()
        )["source_revision"],
        "license_included": licensed,
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


def scan_output(staging: Path) -> None:
    problems = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        for problem in scan_text(path):
            problems.append(f"{path.relative_to(staging)}: {problem}")
    if problems:
        raise RuntimeError("sanitized bundle scan failed:\n" + "\n".join(problems))


def write_deterministic_archive(bundle: Path, archive: Path) -> None:
    """Write a byte-reproducible tar.gz plus a conventional checksum file."""
    checksum = archive.with_name(f"{archive.name}.sha256")
    if archive.exists() or checksum.exists():
        raise RuntimeError(f"refusing to overwrite archive or checksum: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in (bundle, *sorted(bundle.rglob("*"))):
                    arcname = Path(bundle.name) / path.relative_to(bundle)
                    info = tar.gettarinfo(str(path), arcname.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_dir():
                        info.mode = 0o755
                        tar.addfile(info)
                    else:
                        info.mode = 0o644
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--allow-unlicensed-preview", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    archive = args.archive.resolve() if args.archive else None
    if archive and archive.suffixes[-2:] != [".tar", ".gz"]:
        raise RuntimeError("--archive must end in .tar.gz")
    licensed = (repo / "LICENSE").is_file()
    if not licensed and not args.allow_unlicensed_preview:
        raise RuntimeError(
            "public bundle blocked: add an approved project LICENSE first; "
            "use --allow-unlicensed-preview only for local review"
        )

    files = source_files(repo)
    preflight(repo, files)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copy_sanitized(repo, staging)
        write_bundle_metadata(repo, staging, licensed)
        scan_output(staging)
        staging.replace(output)
        if archive:
            write_deterministic_archive(output, archive)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"release bundle written to {output}")
    if archive:
        print(f"release archive written to {archive}")
        print(f"archive sha256: {sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
