"""Training helpers for satellite late-fusion models."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cloud2watt.training import masked_mae_loss


def _forward(model: nn.Module, batch: dict[str, object], device: torch.device,
             perturbation: str = "none") -> torch.Tensor:
    return model(
        batch["satellite"].to(device), batch["satellite_mask"].to(device),
        batch["power_history"].to(device), batch["solar_future"].to(device),
        batch["site"].to(device), satellite_perturbation=perturbation,
    )


def run_satellite_epoch(model: nn.Module, loader: DataLoader, *, device: torch.device,
                        optimizer: torch.optim.Optimizer | None = None,
                        use_amp: bool = False, accumulation_steps: int = 1) -> float:
    """Run one masked-MAE epoch with optional CUDA AMP and gradient accumulation."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    training = optimizer is not None
    model.train(training)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    total_loss, total_targets = 0.0, 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, batch in enumerate(loader, start=1):
            amp_context = torch.autocast(device_type="cuda") if use_amp else nullcontext()
            with amp_context:
                prediction = _forward(model, batch, device)
                target = batch["target"].to(device)
                mask = batch["target_mask"].to(device).bool()
                loss = masked_mae_loss(prediction, target, mask)
            if optimizer is not None:
                (loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            count = int(mask.sum())
            total_loss += float(loss.detach()) * count
            total_targets += count
    if total_targets == 0:
        raise ValueError("epoch contains no valid targets")
    return total_loss / total_targets


def predict_satellite(model: nn.Module, loader: DataLoader, device: torch.device, *,
                      perturbation: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict one partition, optionally zeroing or deterministically shuffling satellite input."""
    model.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(_forward(model, batch, device, perturbation).cpu().numpy())
            targets.append(batch["target"].numpy())
            masks.append(batch["target_mask"].numpy().astype(bool))
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
