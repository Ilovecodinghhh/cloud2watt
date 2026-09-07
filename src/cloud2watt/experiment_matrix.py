"""Utilities for safely restarting formal experiment matrices."""

from __future__ import annotations

import json
from pathlib import Path


def completed_formal_run(output_root: Path, run_name: str, mode: str, seed: int,
                         max_epochs: int) -> bool:
    """Return whether a matching uncapped, locked-test run has final metrics."""
    for path in output_root.glob(f"{run_name}-{mode}-s{seed}-*"):
        provenance_path = path / "provenance.json"
        if not provenance_path.exists() or not (path / "metrics.json").exists():
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            provenance.get("mode") == mode
            and provenance.get("seed") == seed
            and provenance.get("max_epochs") == max_epochs
            and provenance.get("max_train_samples") is None
            and provenance.get("max_validation_samples") is None
            and provenance.get("test_unlocked") is False
        ):
            return True
    return False
