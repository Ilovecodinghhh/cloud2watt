"""Spatial extraction helpers for satellite arrays."""

from __future__ import annotations

import numpy as np


def nearest_index(coordinates: np.ndarray, value: float) -> int:
    """Return the nearest finite coordinate index for ascending or descending axes."""
    axis = np.asarray(coordinates)
    if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
        raise ValueError("coordinates must be a non-empty finite 1-D array")
    return int(np.abs(axis - value).argmin())


def center_crop(
    array: np.ndarray,
    *,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    center_x: float,
    center_y: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Extract an exact site-centred crop from arrays ending in ``[..., y, x]``."""
    if array.ndim < 2 or height < 1 or width < 1:
        raise ValueError("array must be at least 2-D and crop dimensions must be positive")
    if array.shape[-2:] != (len(y_coordinates), len(x_coordinates)):
        raise ValueError("coordinate lengths must match the last two array dimensions")

    center_row = nearest_index(y_coordinates, center_y)
    center_column = nearest_index(x_coordinates, center_x)
    row_start = center_row - height // 2
    column_start = center_column - width // 2
    row_stop = row_start + height
    column_stop = column_start + width
    outside_domain = (
        row_start < 0
        or column_start < 0
        or row_stop > array.shape[-2]
        or column_stop > array.shape[-1]
    )
    if outside_domain:
        raise ValueError("requested crop extends beyond the satellite domain")
    return array[..., row_start:row_stop, column_start:column_stop]
