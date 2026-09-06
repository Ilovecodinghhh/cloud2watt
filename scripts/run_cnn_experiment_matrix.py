"""Run the frozen three-mode, three-seed CNN experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_late_fusion.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_epochs = int(config["training"]["max_epochs"])
    for mode in ("power_solar", "satellite_solar", "full"):
        for seed in config["random_seeds"]:
            if completed_formal_run(
                args.output_root, str(config["run_name"]), mode, int(seed), max_epochs
            ):
                print(f"skipping completed formal run: mode={mode} seed={seed}", flush=True)
                continue
            subprocess.run([
                sys.executable, "scripts/train_cnn_late_fusion.py",
                "--config", str(args.config), "--data", str(args.data),
                "--output-root", str(args.output_root), "--mode", mode,
                "--seed", str(seed),
            ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
