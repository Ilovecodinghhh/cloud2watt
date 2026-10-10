"""Read permitted power windows before materializing labels, retaining row references."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_power_for_samples(path: Path, samples: pd.DataFrame) -> pd.DataFrame:
    """Read key columns globally, labels only inside the requested sites/time envelope.

    Original positional indices are retained. Unread rows have NaN payloads. For
    final evaluation use physically separated manifests/files as well: this helper
    does not turn a mixed file into file-level access control.
    """
    if samples.empty:
        raise ValueError("cannot read power for empty samples")
    metadata = pd.read_parquet(path, columns=["ss_id", "datetime_GMT"])
    keys = pd.MultiIndex.from_frame(metadata)
    if not keys.is_unique:
        raise ValueError("duplicate power keys")
    references = np.concatenate([
        np.concatenate(samples[column].map(np.asarray).to_list()).astype(np.int64)
        for column in ("power_history_row_indices", "target_row_indices")])
    references = np.unique(references[references >= 0])
    if not len(references) or references[-1] >= len(metadata):
        raise ValueError("invalid power row references")
    requested = metadata.iloc[references]
    permitted = pd.read_parquet(path, filters=[
        ("ss_id", "in", requested.ss_id.unique().tolist()),
        ("datetime_GMT", ">=", requested.datetime_GMT.min()),
        ("datetime_GMT", "<=", requested.datetime_GMT.max()),
    ])
    available = pd.MultiIndex.from_frame(permitted[["ss_id", "datetime_GMT"]])
    if not keys[references].isin(available).all():
        raise ValueError("requested power rows absent from filtered read")
    return permitted.set_index(["ss_id", "datetime_GMT"]).reindex(keys).reset_index()
