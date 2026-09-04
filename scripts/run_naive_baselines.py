"""Run leakage-safe naive baselines on a Cloud2Watt paired dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cloud2watt.baselines import SiteTimeClimatology, persistence, smart_persistence
from cloud2watt.data.paired import FORECAST_MINUTES
from cloud2watt.evaluation.metrics import (
    event_precision_recall_f1,
    masked_mae,
    masked_nmae,
    masked_rmse,
    ramp_labels,
    skill_score,
)
from cloud2watt.evaluation.splits import build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--git-commit")
    parser.add_argument(
        "--unlock-test",
        action="store_true",
        help="Include the locked test split in a final report after choices are frozen",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_values(power_values: np.ndarray, indices: object) -> np.ndarray:
    index = np.asarray(indices, dtype=int)
    result = np.full(len(index), np.nan, dtype=float)
    valid = index >= 0
    result[valid] = power_values[index[valid]]
    return result


def build_predictions(
    samples: pd.DataFrame,
    power: pd.DataFrame,
    climatology: SiteTimeClimatology,
    *,
    minimum_ghi: float,
) -> pd.DataFrame:
    values = power["normalized_power"].to_numpy(dtype=float)
    ghi = power["clear_sky_ghi_wm2"].to_numpy(dtype=float)
    records = []
    for row in samples.itertuples(index=False):
        history_indices = np.asarray(row.power_history_row_indices, dtype=int)
        target_indices = np.asarray(row.target_row_indices, dtype=int)
        current_index = int(history_indices[-1])
        targets = _target_values(values, target_indices)
        target_ghi = _target_values(ghi, target_indices)
        target_times = [
            pd.Timestamp(row.issue_time_utc) + timedelta(minutes=offset)
            for offset in FORECAST_MINUTES
        ]
        records.append(
            {
                "site_id": str(row.site_id),
                "issue_time_utc": row.issue_time_utc,
                "temporal_split": row.temporal_split,
                "site_split": row.site_split,
                "solar_elevation_deg": float(power.iloc[current_index]["solar_elevation_deg"]),
                "observed": targets.tolist(),
                "target_mask": list(row.target_mask),
                "persistence": persistence(values[current_index], len(FORECAST_MINUTES)).tolist(),
                "smart_persistence": smart_persistence(
                    values[current_index],
                    ghi[current_index],
                    target_ghi,
                    minimum_ghi=minimum_ghi,
                ).tolist(),
                "climatology": climatology.predict(str(row.site_id), target_times).tolist(),
                "current_power": float(values[current_index]),
            }
        )
    return pd.DataFrame.from_records(records)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = ("persistence", "smart_persistence", "climatology")
    for split in sorted(predictions["evaluation_partition"].unique()):
        split_frame = predictions.loc[predictions["evaluation_partition"].eq(split)]
        truth = np.stack(split_frame["observed"])
        mask = np.stack(split_frame["target_mask"]).astype(bool)
        persistence_values = np.stack(split_frame["persistence"])
        for horizon_index, horizon in enumerate(FORECAST_MINUTES):
            baseline_mae = masked_mae(
                truth[:, horizon_index],
                persistence_values[:, horizon_index],
                mask[:, horizon_index],
            )
            for model in models:
                forecast = np.stack(split_frame[model])
                model_mae = masked_mae(
                    truth[:, horizon_index], forecast[:, horizon_index], mask[:, horizon_index]
                )
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "horizon_minutes": horizon,
                        "mae": model_mae,
                        "rmse": masked_rmse(
                            truth[:, horizon_index],
                            forecast[:, horizon_index],
                            mask[:, horizon_index],
                        ),
                        "nmae": masked_nmae(
                            truth[:, horizon_index],
                            forecast[:, horizon_index],
                            mask[:, horizon_index],
                        ),
                        "skill_score": 0.0
                        if model == "persistence"
                        else skill_score(model_mae, baseline_mae),
                        "valid_targets": int(mask[:, horizon_index].sum()),
                    }
                )
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["issue_time_utc"], utc=True).dt.strftime("%Y-%m-%d")
    frame["solar_elevation_band"] = pd.cut(
        frame["solar_elevation_deg"],
        bins=[-90, 0, 15, 30, 90],
        labels=["night", "low", "medium", "high"],
    ).astype(str)
    for grouping in ("site_id", "date", "solar_elevation_band"):
        for (split, key), group in frame.groupby(
            ["evaluation_partition", grouping], observed=True
        ):
            truth = np.stack(group["observed"])
            mask = np.stack(group["target_mask"]).astype(bool)
            for model in ("persistence", "smart_persistence", "climatology"):
                forecast = np.stack(group[model])
                try:
                    mae = masked_mae(truth, forecast, mask)
                    rmse = masked_rmse(truth, forecast, mask)
                except ValueError:
                    continue
                records.append(
                    {
                        "split": str(split),
                        "grouping": grouping,
                        "group": str(key),
                        "model": model,
                        "mae": mae,
                        "rmse": rmse,
                        "valid_targets": int(mask.sum()),
                    }
                )
    return records


def summary_metrics(
    predictions: pd.DataFrame, grouped: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Report sample-weighted overall and site-macro errors."""
    summaries = []
    site_metrics = pd.DataFrame(grouped)
    site_metrics = site_metrics.loc[site_metrics["grouping"].eq("site_id")]
    for partition, group in predictions.groupby("evaluation_partition"):
        truth = np.stack(group["observed"])
        mask = np.stack(group["target_mask"]).astype(bool)
        for model in ("persistence", "smart_persistence", "climatology"):
            forecast = np.stack(group[model])
            macro = site_metrics.loc[
                site_metrics["split"].eq(partition) & site_metrics["model"].eq(model)
            ]
            summaries.append(
                {
                    "split": partition,
                    "model": model,
                    "weighted_mae": masked_mae(truth, forecast, mask),
                    "weighted_rmse": masked_rmse(truth, forecast, mask),
                    "macro_site_mae": float(macro["mae"].mean()),
                    "macro_site_rmse": float(macro["rmse"].mean()),
                }
            )
    return summaries


def ramp_report(predictions: pd.DataFrame, config: dict[str, object]) -> dict[str, object]:
    report: dict[str, object] = {}
    thresholds = [float(config["ramp"]["primary_threshold"])]
    if predictions["temporal_split"].eq("validation").any():
        thresholds += [
            float(value) for value in config["ramp"]["validation_sensitivity_thresholds"]
        ]
    for split, split_frame in predictions.groupby("evaluation_partition"):
        split_result = {}
        truth = np.stack(split_frame["observed"])
        mask_30 = np.stack(split_frame["target_mask"])[:, 1].astype(bool)
        for threshold in sorted(set(thresholds)):
            if not split.startswith("validation_") and threshold != float(
                config["ramp"]["primary_threshold"]
            ):
                continue
            observed = ramp_labels(split_frame["current_power"].to_numpy(), truth[:, 1], threshold)
            threshold_result = {}
            for model in ("persistence", "smart_persistence", "climatology"):
                forecast_30 = np.stack(split_frame[model])[:, 1]
                predicted = ramp_labels(
                    split_frame["current_power"].to_numpy(), forecast_30, threshold
                )
                model_result = {}
                for direction, name in ((1, "up"), (-1, "down")):
                    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
                    for _site_id, positions in split_frame.groupby("site_id").indices.items():
                        positions = np.asarray(positions, dtype=int)
                        valid = positions[mask_30[positions]]
                        score = event_precision_recall_f1(
                            split_frame.iloc[valid]["issue_time_utc"],
                            observed[valid],
                            predicted[valid],
                            direction=direction,
                            tolerance_minutes=int(config["ramp"]["tolerance_minutes"]),
                        )
                        for key in totals:
                            totals[key] += int(score[key])
                    tp, fp, fn = (
                        totals["true_positive"],
                        totals["false_positive"],
                        totals["false_negative"],
                    )
                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    model_result[name] = totals | {
                        "precision": precision,
                        "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall)
                        if precision + recall
                        else 0.0,
                    }
                threshold_result[model] = model_result
            split_result[f"{threshold:.2f}"] = threshold_result
        report[str(split)] = split_result
    return report


def plot_horizon_metrics(metrics: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for (split, model), group in metrics.groupby(["split", "model"]):
        axis.plot(group["horizon_minutes"], group["mae"], marker="o", label=f"{split}/{model}")
    axis.set(xlabel="Forecast horizon (minutes)", ylabel="Masked MAE", title="Naive baselines")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_manifest_path = args.data / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if data_manifest["dataset_version"] != config["data_version"]:
        raise SystemExit("data version does not match baseline config")
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    sites = pd.read_parquet(args.data / "sites.parquet")
    power = pd.read_parquet(args.data / "power_15min.parquet")
    assignments, split_manifest = build_split_manifest(
        samples,
        sites,
        data_manifest_hash=data_manifest["manifest_content_sha256"],
        seed=int(config["random_seed"]),
        embargo_hours=int(config["embargo_hours"]),
    )
    training = assignments.loc[
        assignments["temporal_split"].eq("train") & assignments["site_split"].eq("development")
    ]
    training_indices = sorted(
        {
            int(index)
            for row in training.itertuples(index=False)
            for index in list(row.power_history_row_indices) + list(row.target_row_indices)
            if int(index) >= 0
        }
    )
    climatology = SiteTimeClimatology().fit(power.iloc[training_indices])
    requested_splits = ["validation"] + (["test"] if args.unlock_test else [])
    evaluation_samples = assignments.loc[assignments["temporal_split"].isin(requested_splits)]
    predictions = build_predictions(
        evaluation_samples,
        power,
        climatology,
        minimum_ghi=float(config["smart_persistence"]["minimum_clear_sky_ghi_wm2"]),
    )
    predictions["evaluation_partition"] = (
        predictions["temporal_split"].astype(str)
        + "_"
        + predictions["site_split"].astype(str)
    )
    metrics = metric_rows(predictions)
    git_commit = (
        args.git_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    identity = {
        "git_commit": git_commit,
        "config_hash": sha256_file(args.config),
        "data_manifest_hash": data_manifest["manifest_content_sha256"],
        "split_manifest_hash": split_manifest["split_manifest_sha256"],
        "test_unlocked": args.unlock_test,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    run_id = f"{config['run_name']}-{identity_hash[:12]}"
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "plots").mkdir()
    (output / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    assignments.to_parquet(output / "split_assignments.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics.to_csv(output / "metrics_by_horizon.csv", index=False)
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    grouped = grouped_metrics(predictions)
    metrics_payload = {
        "by_horizon": metrics.to_dict(orient="records"),
        "summary": summary_metrics(predictions, grouped),
        "by_group": grouped,
        "ramps": ramp_report(predictions, config),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = identity | {
        "run_id": run_id,
        "split_ranges": split_manifest["boundaries"],
        "site_split_hash": hashlib.sha256(
            "\n".join(split_manifest["holdout_site_ids"]).encode()
        ).hexdigest(),
        "random_seeds": [int(config["random_seed"])],
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "packages": {name: version(name) for name in ("numpy", "pandas", "pvlib", "pyarrow")},
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_horizon_metrics(metrics, output / "plots" / "mae_by_horizon.png")
    print(f"baseline run written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
