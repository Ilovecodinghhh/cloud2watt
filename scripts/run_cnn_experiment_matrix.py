"""Run the frozen three-mode, three-seed CNN experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_late_fusion.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for mode in ("power_solar", "satellite_solar", "full"):
        for seed in config["random_seeds"]:
            subprocess.run([
                sys.executable, "scripts/train_cnn_late_fusion.py",
                "--config", str(args.config), "--data", str(args.data),
                "--output-root", str(args.output_root), "--mode", mode,
                "--seed", str(seed),
            ], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
