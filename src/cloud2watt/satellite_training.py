"""Compatibility helpers backed by the shared, corrected epoch implementation."""
from __future__ import annotations

import numpy as np
import torch

from cloud2watt.epoch import forward_batch, train_epoch


def run_satellite_epoch(model, loader, *, device, optimizer=None, use_amp=False,
                        accumulation_steps=1, scaler=None):
    return train_epoch(model, loader, device=device, optimizer=optimizer, satellite=True,
                       use_amp=use_amp, accumulation_steps=accumulation_steps, scaler=scaler)


def predict_satellite(model, loader, device, *, perturbation="none"):
    model.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(forward_batch(model, batch, device, True, perturbation)
                               .float().cpu().numpy())
            targets.append(batch["target"].numpy())
            masks.append(batch["target_mask"].numpy().astype(bool))
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
