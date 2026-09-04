from pathlib import Path

import pytest

from scripts.build_30d_dataset import annual_storage_estimates, validate_new_version_path


def test_version_path_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "paired-30d-v1"
    assert validate_new_version_path(output, "paired-30d-v1") == output
    output.mkdir()
    with pytest.raises(FileExistsError):
        validate_new_version_path(output, "paired-30d-v1")
    with pytest.raises(ValueError):
        validate_new_version_path(tmp_path / "wrong", "paired-30d-v1")


def test_storage_estimate_scales_by_days_and_sites() -> None:
    result = annual_storage_estimates(30_000, 30, 20)
    assert result["100_sites_one_year_bytes"] == 1_825_000
    assert result["300_sites_one_year_bytes"] == 5_475_000
