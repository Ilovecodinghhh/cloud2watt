"""Audit saved development predictions without reading historical test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from cloud2watt.evaluation.forecast import score_forecast
from cloud2watt.run_state import atomic_json, file_hash, is_complete

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def audit(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "preregistration.json", {
        "stage": "E00", "route": "B", "project_version": "2.0",
        "input": str(root), "test_access": "forbidden", "training_runs": 0,
        "scope": "saved validation predictions only", "seeds": [42, 123, 2026],
    })
    runs, groups, identities = [], [], {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for directory in sorted(root.iterdir()):
        if not (directory / "complete.json").exists():
            continue
        identity = json.loads((directory / "provenance.json").read_text("utf-8"))["identity"]
        if not is_complete(directory, identity):
            raise ValueError(f"artifact verification failed: {directory.name}")
        if identity["test_unlocked"]:
            raise ValueError("E00 requires development-only runs")
        metrics = json.loads((directory / "metrics.json").read_text("utf-8"))
        history = pd.read_csv(directory / "training_history.csv")
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        mode, seed = identity["mode"], identity["seed"]
        if mode not in {"mlp", "power_solar", "full"}:
            raise ValueError(f"unexpected mode {mode}")
        best = history.loc[history.validation_daylight_primary_mae.idxmin()]
        if int(best.epoch) != checkpoint["epoch"]:
            raise ValueError("best checkpoint/history mismatch")
        axes[["mlp", "power_solar", "full"].index(mode)].plot(
            history.epoch, history.validation_daylight_primary_mae * 100, label=str(seed))
        result = {"run": directory.name, "mode": mode, "seed": seed,
                  "artifact_verified": True, "device": identity["device"],
                  "best_epoch": int(best.epoch), "epochs": len(history),
                  "recorded_source_hash": identity["source_hash"], "partitions": {}}
        for partition in ("validation_development", "validation_holdout"):
            frame = pd.read_parquet(directory / "predictions.parquet", filters=[
                ("partition", "==", partition), ("perturbation", "==", "none")])
            keys = frame[["site_id", "issue_time_utc"]]
            pred, obs, mask, elevation = [np.stack(frame[k]) for k in (
                "prediction", "observed", "target_mask", "target_solar_elevation_deg")]
            report = score_forecast(pred, obs, mask, elevation, keys)
            np.testing.assert_allclose(report["daylight_primary_mae"],
                metrics[partition]["none"]["daylight_primary_mae"], rtol=1e-7, atol=1e-9)
            if partition in identities and report["comparison"] != identities[partition]:
                raise ValueError("models do not share the same evaluation population")
            identities[partition] = report["comparison"]
            result["partitions"][partition] = {
                "mae": report["daylight_primary_mae"],
                "issue_dates": sorted(pd.to_datetime(frame.issue_time_utc).dt.date
                                      .astype(str).unique().tolist()),
                "rows": len(frame), "sites": frame.site_id.nunique(),
                "daylight_counts": (mask.astype(bool) & (elevation > 5)).sum(0).tolist(),
                "valid_counts": mask.astype(bool).sum(0).tolist(),
            }
            for kind, group_key in (("date", "by_issue_date"), ("site", "by_site")):
                for key, values in report.get(group_key, {}).items():
                    groups.append({"mode": mode, "seed": seed, "partition": partition,
                                   "kind": kind, "key": key,
                                   "mae": values["daylight_primary_mae"]})
            for row in report["daylight"]:
                groups.append({"mode": mode, "seed": seed, "partition": partition,
                               "kind": "horizon", "key": str(row["horizon_minutes"]),
                               "mae": row["mae"]})
        runs.append(result)
    if {(r["mode"], r["seed"]) for r in runs} != {
            (mode, seed) for mode in ("mlp", "power_solar", "full")
            for seed in (42, 123, 2026)} or len(runs) != 9:
        raise ValueError("expected exactly the nine compact runs")
    for ax, mode in zip(axes, ("mlp", "power_solar", "full"), strict=True):
        ax.set(title=mode, xlabel="Epoch", ylabel="Validation daylight MAE (capacity pp)")
        ax.legend(title="Seed")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "validation_learning_curves.png", dpi=150)
    plt.close(fig)
    grouped = pd.DataFrame(groups)
    grouped.to_csv(output / "grouped_metrics.csv", index=False, encoding="utf-8")
    table = grouped.groupby(["partition", "kind", "key", "mode"]).mae.mean().unstack()
    table["full_minus_mlp"] = table.full - table.mlp
    table["full_minus_power_solar"] = table.full - table.power_solar
    table.to_csv(output / "paired_group_differences.csv", encoding="utf-8")
    summary = {"stage": "E00", "status": "complete", "runs": runs,
               "historical_protocol": "c2w-eval-v2.1", "test_labels_read": False,
               "test_scored": False, "new_training_runs": 0,
               "ci_status": "descriptive only: fewer than 20 independent issue dates",
               "script_sha256": file_hash(Path(__file__)),
               "curve_warning": "train_masked_mae and validation primary use different objectives"}
    atomic_json(output / "audit.json", summary)
    atomic_json(output / "complete.json", {"audit_sha256": file_hash(output / "audit.json")})
    print(json.dumps({"verified_runs": len(runs), "output": str(output)}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/compact-route-v21/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.runs, args.output)
