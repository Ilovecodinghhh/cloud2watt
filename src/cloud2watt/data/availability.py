"""Explicit archival observation times and hypothetical publication delays."""
from __future__ import annotations

import numpy as np
import pandas as pd


def availability_contract(times, *, delay_minutes: int, basis: str = "scenario"):
    """Return a ledger; a scenario delay is never represented as measured latency."""
    if delay_minutes < 0 or basis not in {"scenario", "measured"}:
        raise ValueError("nonnegative delay and explicit availability basis required")
    if basis == "measured":
        raise ValueError("measured availability requires source publication timestamps")
    observation = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if observation.has_duplicates or not observation.is_monotonic_increasing:
        raise ValueError("observation times must be unique and increasing")
    return pd.DataFrame({
        "observation_time": observation,
        "available_time": observation + pd.Timedelta(minutes=delay_minutes),
        "availability_basis": basis,
    })


def delayed_history_indices(issue_times, ledger, *, delay_minutes: int):
    """Select exact delayed slots; missing slots stay -1, never future-filled.

    Targets and power histories are unchanged. This function models satellite
    delay only; callers must explicitly adopt the shifted model input contract.
    """
    if delay_minutes < 0:
        raise ValueError("delay must be nonnegative")
    observation = pd.DatetimeIndex(ledger.observation_time)
    available = pd.DatetimeIndex(ledger.available_time)
    if (observation.has_duplicates or not observation.is_monotonic_increasing
            or observation.hasnans or available.hasnans or (available < observation).any()):
        raise ValueError("invalid observation/availability ledger")
    issues = pd.DatetimeIndex(pd.to_datetime(issue_times, utc=True))
    selected = np.full((len(issues), 4), -1, dtype=np.int64)
    for column, offset in enumerate((-45, -30, -15, 0)):
        desired = issues + pd.Timedelta(minutes=offset - delay_minutes)
        indices = observation.get_indexer(desired)
        valid = indices >= 0
        valid[valid] &= available[indices[valid]] <= issues[valid]
        selected[valid, column] = indices[valid]
    return selected
