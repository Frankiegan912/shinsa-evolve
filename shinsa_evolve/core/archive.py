"""Append-only candidate archive (archive.jsonl) with dual-mode ranking.

Every candidate stores both its claimed score and its audited score; the run's
selection mode only decides which column drives parent selection. Rankings can
therefore be recomputed under either mode after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path


class Archive:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "archive.jsonl"

    def load(self) -> list[dict]:
        records = []
        if not self.path.exists():
            return records
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A partial trailing line from a killed run is tolerated.
                    continue
        return records

    def append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def done(self) -> list[dict]:
        return [r for r in self.load() if r.get("status") == "done"]

    def count_done(self) -> int:
        return len(self.done())

    def next_id(self) -> int:
        records = self.load()
        return 1 + max((r.get("id", -1) for r in records), default=-1)

    def top(self, mode: str, k: int = 5) -> list[dict]:
        key = "claimed" if mode == "claimed" else "audited"
        return sorted(self.done(), key=lambda r: r.get(key, -1e18), reverse=True)[:k]
