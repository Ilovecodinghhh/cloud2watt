"""Immutable dataset-version and storage-estimation helpers."""

from __future__ import annotations

from pathlib import Path


def annual_storage_estimates(total_bytes: int, days: int, sites: int) -> dict[str, int]:
    """Linearly extrapolate measured derived bytes to annual site counts."""
    if total_bytes < 0 or days <= 0 or sites <= 0:
        raise ValueError("bytes must be non-negative; days and sites must be positive")
    return {
        f"{target_sites}_sites_one_year_bytes": round(
            total_bytes * 365 / days * target_sites / sites
        )
        for target_sites in (100, 300)
    }


def validate_new_version_path(output: Path, dataset_version: str) -> Path:
    """Return a new immutable version path after validating its name and absence."""
    if output.name != dataset_version:
        raise ValueError("output directory name must equal config dataset_version")
    if output.exists():
        raise FileExistsError(f"immutable dataset version already exists: {output}")
    return output
