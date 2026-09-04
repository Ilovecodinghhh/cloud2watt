"""Versioned local storage for derived mini datasets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from cloud2watt.data.schema import ForecastSample


def write_derived_dataset(
    output_dir: Path,
    samples: Sequence[ForecastSample],
    *,
    manifest: dict[str, object],
) -> None:
    """Write tensors to Zarr and scalar sample metadata to Parquet."""
    if not samples:
        raise ValueError("at least one sample is required")
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {
        name: np.stack([getattr(sample, name) for sample in samples])
        for name in (
            "satellite",
            "power_history",
            "solar_features",
            "site_features",
            "target_power",
            "target_mask",
        )
    }
    root = zarr.open_group(output_dir / "samples.zarr", mode="w")
    for name, values in arrays.items():
        root.create_array(name, data=values, chunks=(1, *values.shape[1:]))

    pd.DataFrame(sample.index_record() for sample in samples).to_parquet(
        output_dir / "sample_index.parquet", index=False
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
