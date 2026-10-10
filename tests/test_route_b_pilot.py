"""Engineering pilots cannot open locked periods or cross temporal support."""

import runpy
from pathlib import Path

import pandas as pd
import pytest

bounds = runpy.run_path(str(Path(__file__).parents[1] / "scripts/build_route_b_pilot.py"))[
    "pilot_bounds"
]


def test_locked_regions_and_times_refused():
    plan = {
        "regions": [
            {"region_id": "dev", "role": "development"},
            {"region_id": "test", "role": "locked_geography"},
        ],
        "windows": [
            {
                "id": "dev",
                "role": "development",
                "start": "2021-01-01T00:00Z",
                "end": "2021-01-29T00:00Z",
            },
            {
                "id": "test",
                "role": "locked_final",
                "start": "2022-01-01T00:00Z",
                "end": "2022-02-26T00:00Z",
            },
        ],
    }
    for region, window in (("test", "dev"), ("dev", "test")):
        with pytest.raises(ValueError, match="locked"):
            bounds(plan, region, window, "2021-01-01T00:00Z", "2021-01-03T00:00Z")
    with pytest.raises(ValueError, match="outside"):
        bounds(plan, "dev", "dev", "2020-12-31T00:00Z", "2021-01-02T00:00Z")
    _, start, end, issues = bounds(plan, "dev", "dev", "2021-01-01T00:00Z", "2021-01-03T00:00Z")
    assert (issues - pd.Timedelta(hours=1) >= start).all()
    assert (issues + pd.Timedelta(hours=4) < end).all()
