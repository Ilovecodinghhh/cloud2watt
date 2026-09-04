"""Deterministic solar-position and clear-sky features."""

from __future__ import annotations

import pandas as pd
from pvlib.location import Location


def build_solar_features(times: pd.DatetimeIndex, sites: pd.DataFrame) -> pd.DataFrame:
    """Compute per-site features using only UTC time and rounded coordinates."""
    if times.tz is None:
        raise ValueError("times must be timezone-aware")
    records = []
    for site in sites.itertuples(index=False):
        location = Location(
            latitude=float(site.latitude_rounded),
            longitude=float(site.longitude_rounded),
            tz="UTC",
        )
        position = location.get_solarposition(times)
        clear_sky = location.get_clearsky(times, model="ineichen")
        records.append(
            pd.DataFrame(
                {
                    "ss_id": site.ss_id,
                    "datetime_GMT": times,
                    "solar_elevation_deg": position["apparent_elevation"].to_numpy(),
                    "solar_azimuth_deg": position["azimuth"].to_numpy(),
                    "clear_sky_ghi_wm2": clear_sky["ghi"].to_numpy(),
                }
            )
        )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
