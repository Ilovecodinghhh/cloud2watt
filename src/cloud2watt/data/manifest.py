"""Deterministic mini-dataset manifest construction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta


def build_fixture_manifest(
    site_ids: Sequence[str],
    *,
    start_utc: datetime,
    days: int = 7,
    interval_minutes: int = 15,
) -> dict[str, object]:
    """Build a reproducible selection manifest without embedding source data."""
    if start_utc.tzinfo is None or start_utc.utcoffset() != timedelta(0):
        raise ValueError("start_utc must be timezone-aware UTC")
    if len(site_ids) != len(set(site_ids)) or not site_ids:
        raise ValueError("site_ids must be non-empty and unique")
    if days < 1 or interval_minutes < 1:
        raise ValueError("days and interval_minutes must be positive")
    end_utc = start_utc + timedelta(days=days)
    return {
        "schema_version": 1,
        "site_ids": list(site_ids),
        "site_count": len(site_ids),
        "start_utc": start_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "end_utc_inclusive": end_utc.isoformat().replace("+00:00", "Z"),
        "days": days,
        "interval_minutes": interval_minutes,
        "issue_times_per_site": days * 24 * 60 // interval_minutes,
    }
