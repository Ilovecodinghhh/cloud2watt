"""Masked training with target-weighted accumulation and epoch checkpoints."""
from __future__ import annotations

from contextlib import nullcontext
from itertools import islice

import numpy as np
import pandas as pd
import torch

from cloud2watt.evaluation.forecast import require_primary, score_forecast, validate_forecast
from cloud2watt.run_state import (
    CHECKPOINT_VERSION,
    atomic_path,
    atomic_torch,
    capture_rng,
    load_epoch,
    restore_rng,
)
from cloud2watt.training import masked_mae_loss


def forward_batch(model, batch, device, satellite=False, perturbation="none"):
    inputs = [batch[k].to(device) for k in ("power_history", "solar_future", "site")]
    if satellite:
        return model(batch["satellite"].to(device), batch["satellite_mask"].to(device),
                     *inputs, satellite_perturbation=perturbation)
    return model(*inputs)


def train_epoch(model, loader, *, device, optimizer=None, satellite=False,
                accumulation_steps=1, scaler=None, use_amp=False, primary_loss_weight=0.0):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if use_amp and device.type != "cuda":
        raise ValueError("AMP requires a CUDA device")
    if use_amp and scaler is None:
        raise ValueError("AMP training requires a persistent GradScaler")
    if not 0 <= primary_loss_weight < 1:
        raise ValueError("primary_loss_weight must be in [0, 1) to supervise all outputs")
    model.train(optimizer is not None)
    iterator = iter(loader)
    total, count = 0.0, 0
    context = torch.enable_grad if optimizer is not None else torch.no_grad
    with context():
        while group := list(islice(iterator, accumulation_steps)):
            denominator = sum(int(b["target_mask"].bool().sum()) for b in group)
            if not denominator:
                continue
            day_counts = torch.zeros(3, dtype=torch.int64)
            if primary_loss_weight:
                for batch in group:
                    daylight = (batch["target_mask"].bool()
                                & (batch["target_solar_elevation_deg"] > 5))
                    day_counts += daylight[:, [1, 2, 3]].sum(0).cpu()
            supported = int((day_counts > 0).sum())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for batch in group:
                n = int(batch["target_mask"].bool().sum())
                if not n:
                    continue
                with torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext():
                    prediction = forward_batch(model, batch, device, satellite)
                    target, mask = batch["target"].to(device), batch["target_mask"].to(device)
                    # Include sample keys in failure diagnostics before computing the loss.
                    keys = pd.DataFrame({k: batch[k] for k in ("site_id", "issue_time_utc")}) \
                        if "site_id" in batch else None
                    validate_forecast(prediction.detach().float().cpu().numpy(),
                                      target.cpu().numpy(), mask.cpu().numpy(), keys)
                    loss = masked_mae_loss(prediction, target, mask)
                if optimizer is not None:
                    weighted = loss * (n / denominator) * (1 - primary_loss_weight)
                    if primary_loss_weight and supported:
                        elevation = batch["target_solar_elevation_deg"].to(device)
                        daylight = mask.bool() & (elevation > 5)
                        for column, horizon in enumerate((1, 2, 3)):
                            valid = daylight[:, horizon]
                            if valid.any():
                                error = torch.abs(prediction[valid, horizon].float()
                                                  - target[valid, horizon].float()).sum()
                                weighted = weighted + error * (
                                    primary_loss_weight / (int(day_counts[column]) * supported))
                    if scaler is not None:
                        scaler.scale(weighted).backward()
                    else:
                        weighted.backward()
                total += float(loss.detach()) * n
                count += n
            if optimizer is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
    if not count:
        raise ValueError("epoch contains no valid targets")
    return total / count


def evaluate(model, loader, device, *, satellite=False, perturbation="none", groups=True):
    model.eval()
    predictions, targets, masks, elevations, key_frames = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(forward_batch(model, batch, device, satellite, perturbation)
                               .float().cpu().numpy())
            targets.append(batch["target"].numpy())
            masks.append(batch["target_mask"].numpy().astype(bool))
            elevations.append(batch["target_solar_elevation_deg"].numpy())
            key_frames.append(pd.DataFrame({k: batch[k] for k in ("site_id", "issue_time_utc")}))
    if not predictions:
        raise ValueError("evaluation partition is empty")
    prediction, target, mask, elevation = map(np.concatenate,
                                             (predictions, targets, masks, elevations))
    keys = pd.concat(key_frames, ignore_index=True)
    report = score_forecast(prediction, target, mask, elevation, keys, groups=groups)
    frame = keys.assign(prediction=list(prediction), observed=list(target), target_mask=list(mask),
                        target_solar_elevation_deg=list(elevation))
    return report, frame


def fit_epochs(model, optimizer, loaders, *, device, identity, output, max_epochs,
               patience, satellite=False, accumulation_steps=1, use_amp=False,
               resume=False, stop_after_epoch=None, min_epochs=0, primary_loss_weight=0.0):
    """latest.pt is the atomic epoch transaction, including the best model so far."""
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if not 0 <= min_epochs <= max_epochs:
        raise ValueError("min_epochs must be between zero and max_epochs")
    best_score, best_epoch, stale, history, first = None, 0, 0, [], 1
    best_state = None
    if resume:
        payload = load_epoch(output / "latest.pt", identity)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        restore_rng(payload["rng"], loaders)
        best_score, best_epoch = payload["best_score"], payload["best_epoch"]
        stale, history, first = payload["stale_epochs"], payload["history"], payload["epoch"] + 1
        best_state = payload["best_model_state"]
        if payload["sampler_epoch"] != payload["epoch"]:
            raise ValueError("sampler epoch differs from checkpoint")
    for epoch in range(first, max_epochs + 1):
        if epoch > min_epochs and stale >= patience:
            break
        sampler = loaders["train"].sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        loss = train_epoch(model, loaders["train"], device=device, optimizer=optimizer,
                           satellite=satellite, accumulation_steps=accumulation_steps,
                           scaler=scaler, use_amp=use_amp,
                           primary_loss_weight=primary_loss_weight)
        report, _ = evaluate(model, loaders["validation"], device,
                             satellite=satellite, groups=False)
        score = require_primary(report)
        history.append({"epoch": epoch, "train_masked_mae": loss,
                        "validation_daylight_primary_mae": score})
        if "train_eval" in loaders:
            train_report, _ = evaluate(model, loaders["train_eval"], device,
                                       satellite=satellite, groups=False)
            history[-1]["train_eval_daylight_primary_mae"] = require_primary(train_report)
        if best_score is None or score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        payload = {"checkpoint_version": CHECKPOINT_VERSION, "identity": identity,
                   "model_state": model.state_dict(), "best_model_state": best_state,
                   "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                   "rng": capture_rng(loaders), "epoch": epoch, "sampler_epoch": epoch,
                   "best_epoch": best_epoch, "best_score": best_score, "stale_epochs": stale,
                   "history": history}
        atomic_torch(output / "latest.pt", payload)
        with atomic_path(output / "training_history.csv") as temporary:
            pd.DataFrame(history).to_csv(temporary, index=False, encoding="utf-8")
        print(f"epoch={epoch} train_masked_mae={loss:.6f} "
              f"validation_daylight_primary_mae={score:.6f}", flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return False
    if best_state is None:
        raise ValueError("no eligible best checkpoint")
    model.load_state_dict(best_state)
    atomic_torch(output / "best.pt", {"checkpoint_version": CHECKPOINT_VERSION,
                                     "identity": identity, "epoch": best_epoch,
                                     "selection_score": best_score, "model_state": best_state})
    with atomic_path(output / "training_history.csv") as temporary:
        pd.DataFrame(history).to_csv(temporary, index=False, encoding="utf-8")
    return True
