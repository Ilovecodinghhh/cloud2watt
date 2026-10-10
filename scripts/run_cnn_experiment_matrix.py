"""Run or resume a V2 matrix and rank verified results with the daylight metric."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cloud2watt.experiment import execute, parser_for, prepare
from cloud2watt.experiment_matrix import rank_completed_runs
from cloud2watt.run_state import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_late_fusion.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = []
    for mode in ("power_solar", "satellite_solar", "full"):
        for seed in config["random_seeds"]:
            run_args = parser_for("satellite").parse_args([
                "--config", str(args.config), "--data", str(args.data),
                "--output-root", str(args.output_root), "--mode", mode,
                "--seed", str(seed), "--auto-resume"])
            prepared = prepare("satellite", run_args)
            paths.append(execute(prepared, run_args))
            del prepared
            atomic_json(args.output_root / f"{config['run_name']}-matrix-summary.json",
                        rank_completed_runs(paths, config["random_seeds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
