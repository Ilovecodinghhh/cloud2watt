"""UTC conversion and power-series alignment without future leakage."""

from __future__ import annotations

from typing import Literal

import pandas as pd


def to_utc(values: pd.Series) -> pd.Series:
    """Parse timestamps and normalize them to timezone-aware UTC values."""
    return pd.to_datetime(values, utc=True, errors="raise")


def aggregate_power_15min(
    frame: pd.DataFrame,
    capacities_kwp: pd.Series,
    *,
    value_semantics: Literal["interval_energy", "instantaneous_power"],
    minimum_readings: int = 3,
) -> pd.DataFrame:
    """Convert 5-minute interval energy to 15-minute mean normalized power.

    ``generation_Wh`` is interpreted as energy for the interval ending at
    ``datetime_GMT``. Three readings are summed, divided by 0.25 h, then
    normalized by nominal capacity in watts. Bins use their right edge, so an
    issue time cannot consume a future reading.
    """
    required = {"ss_id", "datetime_GMT", "generation_Wh"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing power columns: {sorted(missing)}")
    if minimum_readings < 1 or minimum_readings > 3:
        raise ValueError("minimum_readings must be between 1 and 3")

    data = frame.loc[:, ["ss_id", "datetime_GMT", "generation_Wh"]].copy()
    data["datetime_GMT"] = to_utc(data["datetime_GMT"])
    data["capacity_kwp"] = data["ss_id"].map(capacities_kwp)
    if data["capacity_kwp"].isna().any() or (data["capacity_kwp"] <= 0).any():
        raise ValueError("every site must have a positive capacity")

    grouped = (
        data.set_index("datetime_GMT")
        .groupby("ss_id")
        .resample("15min", label="right", closed="right", origin="epoch")
    )
    result = grouped.agg(
        generation_Wh=("generation_Wh", "sum"),
        reading_count=("generation_Wh", "count"),
        capacity_kwp=("capacity_kwp", "first"),
    ).reset_index()
    if value_semantics == "interval_energy":
        result["power_w"] = result["generation_Wh"] / 0.25
    elif value_semantics == "instantaneous_power":
        result["power_w"] = result["generation_Wh"] / result["reading_count"]
    else:
        raise ValueError(f"unsupported value_semantics: {value_semantics}")
    result["normalized_power"] = result["power_w"] / (result["capacity_kwp"] * 1000.0)
    result.loc[result["reading_count"] < minimum_readings, "normalized_power"] = float("nan")
    return result.sort_values(["ss_id", "datetime_GMT"], ignore_index=True)
