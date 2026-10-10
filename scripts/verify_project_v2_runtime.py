"""Six bounded synthetic fitting checks and CPU/CUDA numerical parity in Conda."""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cloud2watt.epoch import forward_batch, train_epoch  # noqa: E402
from cloud2watt.models import PowerMLP, SatelliteLateFusion  # noqa: E402
from cloud2watt.run_state import atomic_json, environment_snapshot  # noqa: E402
from cloud2watt.training import masked_mae_loss  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if "raincalib-gpu" not in sys.executable or not torch.cuda.is_available():
        raise ValueError("use the existing raincalib-gpu Conda interpreter with CUDA")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    atomic_json(args.output / "preregistration.json", {
        "stage": "E01-runtime", "synthetic_runs": 6, "steps_per_run": 100,
        "seed": 2026, "parity_atol": 1e-5, "parity_rtol": 1e-4,
        "minimum_loss_reduction_fraction": .5, "test_labels_read": False,
        "python": sys.executable, "torch_path": torch.__file__,
    })
    torch.manual_seed(2026)
    n = 8
    batch = {"power_history": torch.rand(n, 4), "solar_future": torch.rand(n, 6, 3),
             "site": torch.rand(n, 5), "satellite": torch.randn(n, 4, 3, 32, 32),
             "satellite_mask": torch.ones(n, 4, dtype=torch.bool),
             "target_mask": torch.ones(n, 6, dtype=torch.bool),
             "target_solar_elevation_deg": torch.ones(n, 6) * 20}
    batch["target"] = (.2 + .1 * batch["power_history"].mean(1, keepdim=True)
                       + torch.arange(6)[None, :] * .005).expand(n, 6).clone()
    rows = [{key: value[i] for key, value in batch.items()} for i in range(n)]
    loader = DataLoader(rows, batch_size=n)
    results = []
    for mode in ("mlp", "power_solar", "full"):
        satellite = mode != "mlp"
        torch.manual_seed(2026)
        base = PowerMLP(hidden_size=64, dropout=0) if not satellite else SatelliteLateFusion(
            mode=mode, cnn_width=8, embedding_size=32, dropout=0)
        cpu, gpu = copy.deepcopy(base), copy.deepcopy(base).cuda()
        prediction_cpu = forward_batch(cpu, batch, torch.device("cpu"), satellite)
        prediction_gpu = forward_batch(gpu, batch, torch.device("cuda"), satellite)
        torch.testing.assert_close(prediction_cpu, prediction_gpu.cpu(), atol=1e-5, rtol=1e-4)
        masked_mae_loss(prediction_cpu, batch["target"], batch["target_mask"]).backward()
        masked_mae_loss(prediction_gpu, batch["target"].cuda(),
                        batch["target_mask"].cuda()).backward()
        max_gradient_error = 0.
        for a, b in zip(cpu.parameters(), gpu.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=1e-5, rtol=1e-4)
            max_gradient_error = max(max_gradient_error, float((a.grad - b.grad.cpu()).abs().max()))
        for model in (cpu, gpu):
            torch.optim.SGD(model.parameters(), lr=.01).step()
        for a, b in zip(cpu.parameters(), gpu.parameters(), strict=True):
            torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-4)
        runs = []
        for device_name in ("cpu", "cuda"):
            device = torch.device(device_name)
            model = copy.deepcopy(base).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            start = time.perf_counter()
            losses = []
            for _ in range(100):
                losses.append(train_epoch(model, loader, device=device, optimizer=optimizer,
                                          satellite=satellite))
            if device_name == "cuda":
                torch.cuda.synchronize()
            if losses[-1] >= losses[0] * .5:
                raise ValueError(f"synthetic fit did not improve enough: {mode}/{device_name}")
            runs.append({"device": device_name, "first_loss": losses[0], "last_loss": losses[-1],
                         "steps": 100, "seconds": time.perf_counter() - start})
        results.append({"mode": mode, "parity": "passed",
                        "max_gradient_abs_error": max_gradient_error, "runs": runs})
        atomic_json(args.output / "partial.json", {"results": results})
        print(f"{mode}: parity and both 100-step synthetic fits passed", flush=True)
    atomic_json(args.output / "acceptance.json", {
        "status": "passed", "results": results, "environment": environment_snapshot(),
        "scope": "synthetic FP32 parity and fit, not full convergence or AMP accuracy",
        "formal_training_runs": 0, "synthetic_training_runs": 6})


if __name__ == "__main__":
    main()
