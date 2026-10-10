"""Real pilot loader, CPU/CUDA throughput, and deterministic GPU resume checks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cloud2watt.data.satellite_loader import (  # noqa: E402
    SatelliteForecastDataset,
    fit_satellite_statistics,
)
from cloud2watt.epoch import forward_batch, train_epoch  # noqa: E402
from cloud2watt.models import PowerMLP, SatelliteLateFusion  # noqa: E402
from cloud2watt.run_state import atomic_json, environment_snapshot, file_hash  # noqa: E402
from cloud2watt.training import FeatureStatistics, masked_mae_loss  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "raincalib-gpu" not in sys.executable or not torch.cuda.is_available():
        raise ValueError("existing Conda CUDA environment required")
    if not (args.data / "complete.json").exists():
        raise ValueError("pilot must have passed build acceptance")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["data_role"] != "engineering_development_only":
        raise ValueError("this benchmark is restricted to the development pilot")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.output / "preregistration.json",
        {
            "stage": "E02-pilot",
            "manifest_sha256": file_hash(args.data / "manifest.json"),
            "script_sha256": file_hash(Path(__file__)),
            "seed": 2026,
            "batch_size": 128,
            "sample_rule": "evenly spaced pilot indices; fit window only",
            "warmup_steps": 20,
            "timed_steps": 200,
            "resume_continuation_steps": 5,
            "precision": "FP32",
            "test_labels_read": False,
            "amp_forward_absolute_limit": 0.002,
            "amp_training_accepted": False,
            "scope": "200 resident steps plus one complete pilot epoch with I/O",
        },
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026)
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    all_samples = samples.copy()
    samples = samples.iloc[np.linspace(0, len(samples) - 1, min(128, len(samples)), dtype=int)]
    power = pd.read_parquet(args.data / "power_15min.parquet")
    sites = pd.read_parquet(args.data / "sites.parquet")
    features = FeatureStatistics.fit(samples, power, sites)
    satellites = fit_satellite_statistics(
        samples, args.data / "satellite_frames.zarr", maximum_frames=64
    )
    ds = SatelliteForecastDataset(
        samples,
        power,
        sites,
        features,
        satellites,
        args.data / "satellite_frames.zarr",
        spatial_pool=4,
    )
    start = time.perf_counter()
    batch = next(iter(DataLoader(ds, batch_size=128, shuffle=False)))
    load_seconds = time.perf_counter() - start
    batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite model field: {key}")
    results = []
    for mode in ("mlp", "power_solar", "full"):
        torch.manual_seed(2026)
        base = (
            PowerMLP(hidden_size=64, dropout=0)
            if mode == "mlp"
            else SatelliteLateFusion(mode=mode, cnn_width=8, embedding_size=32, dropout=0)
        )
        timings = []
        for device_name, values in (("cpu", batch), ("cuda", batch_gpu)):
            device = torch.device(device_name)
            model = copy.deepcopy(base).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            def step(model=model, optimizer=optimizer, values=values, device=device, mode=mode):
                optimizer.zero_grad(set_to_none=True)
                prediction = forward_batch(model, values, device, mode != "mlp")
                loss = masked_mae_loss(prediction, values["target"], values["target_mask"])
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            for _ in range(20):
                step()
            if device_name == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            losses = [step() for _ in range(200)]
            if device_name == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            timings.append(
                {
                    "device": device_name,
                    "steps": 200,
                    "seconds": elapsed,
                    "samples_per_second": 200 * len(samples) / elapsed,
                    "last_loss": losses[-1],
                    "peak_cuda_allocated": torch.cuda.max_memory_allocated()
                    if device_name == "cuda"
                    else 0,
                }
            )
        epoch_timings = []
        if mode == "full":
            for epoch_device in ("cpu", "cuda"):
                epoch_ds = SatelliteForecastDataset(
                    all_samples,
                    power,
                    sites,
                    features,
                    satellites,
                    args.data / "satellite_frames.zarr",
                    spatial_pool=4,
                )
                epoch_model = copy.deepcopy(base).to(epoch_device)
                epoch_optimizer = torch.optim.Adam(epoch_model.parameters(), lr=0.001)
                if epoch_device == "cuda":
                    torch.cuda.synchronize()
                epoch_start = time.perf_counter()
                epoch_loss = train_epoch(
                    epoch_model,
                    DataLoader(epoch_ds, batch_size=128),
                    device=torch.device(epoch_device),
                    optimizer=epoch_optimizer,
                    satellite=True,
                )
                if epoch_device == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - epoch_start
                epoch_timings.append(
                    {
                        "device": epoch_device,
                        "samples": len(all_samples),
                        "seconds": seconds,
                        "loss": epoch_loss,
                        "samples_per_second": len(all_samples) / seconds,
                    }
                )
        # Same exact model and optimizer state, before the next deterministic batch.
        checkpoint = args.output / f"{mode}-resume.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        reference_losses = [step() for _ in range(5)]
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restored_losses = [step() for _ in range(5)]
        np.testing.assert_array_equal(reference_losses, restored_losses)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        model.eval()
        with torch.no_grad():
            fp32 = forward_batch(model, batch_gpu, torch.device("cuda"), mode != "mlp")
            with torch.autocast("cuda", dtype=torch.float16):
                amp = forward_batch(model, batch_gpu, torch.device("cuda"), mode != "mlp")
        amp_error = float((amp.float() - fp32).abs().max())
        results.append(
            {
                "mode": mode,
                "timings": timings,
                "full_pilot_epoch_with_io": epoch_timings,
                "gpu_optimizer_resume": "exact",
                "amp_forward_max_abs_error": amp_error,
                "amp_forward_pass": bool(amp_error <= 0.002),
                "amp_training_accepted": False,
            }
        )
        atomic_json(args.output / "partial.json", {"results": results})
        print(f"{mode}: throughput and exact GPU optimizer continuation checked", flush=True)
    atomic_json(
        args.output / "acceptance.json",
        {
            "status": "fp32_pilot_passed",
            "results": results,
            "batch_load_seconds": load_seconds,
            "environment": environment_snapshot(),
            "formal_training_runs": 0,
            "limitations": [
                "single two-day pilot; timings are not full-season estimates",
                "augmentation disabled",
                "resume excludes stochastic loader/RNG and interrupted process",
                "AMP forward only, keep training FP32",
                "no validation or final-test score",
            ],
        },
    )


if __name__ == "__main__":
    main()
