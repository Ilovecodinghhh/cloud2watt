import pytest

from cloud2watt.data.probes import (
    ProbeError,
    RemoteFile,
    choose_smallest_file,
    redact_secrets,
    summarize_zarr_metadata,
)


def test_choose_smallest_file_is_bounded_to_prefix_and_suffix() -> None:
    files = [
        RemoteFile("30_minutely/a.parquet", 1),
        RemoteFile("5_minutely/large.parquet", 100),
        RemoteFile("5_minutely/small.parquet", 10),
        RemoteFile("5_minutely/unknown.parquet", None),
    ]

    selected = choose_smallest_file(files, prefix="5_minutely/", suffix=".parquet")

    assert selected.path == "5_minutely/small.parquet"


def test_choose_smallest_file_rejects_empty_match() -> None:
    with pytest.raises(ProbeError, match="No sized file"):
        choose_smallest_file([], prefix="5_minutely/", suffix=".parquet")


def test_summarize_zarr_metadata() -> None:
    payload = {
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            "data/.zarray": {
                "shape": [12, 100, 100, 11],
                "chunks": [12, 100, 100, 11],
                "dtype": "<f2",
                "compressor": {"id": "blosc2"},
            },
            "data/.zattrs": {
                "_ARRAY_DIMENSIONS": ["time", "y", "x", "variable"],
                "platform_name": "Meteosat-10",
                "sensor": "seviri",
                "resolution": 3000.0,
            },
        }
    }

    result = summarize_zarr_metadata(payload)

    assert result["shape"] == [12, 100, 100, 11]
    assert result["compressor"] == {"id": "blosc2"}
    assert result["sensor"] == "seviri"


def test_summarize_zarr_metadata_rejects_inconsistent_dimensions() -> None:
    payload = {
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            "data/.zarray": {"shape": [1, 2], "chunks": [1, 2], "dtype": "<f2"},
            "data/.zattrs": {"_ARRAY_DIMENSIONS": ["time"]},
        }
    }

    with pytest.raises(ProbeError, match="inconsistent"):
        summarize_zarr_metadata(payload)


def test_redact_secrets() -> None:
    message = "Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz and hf_1234567890"

    redacted = redact_secrets(message)

    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "1234567890" not in redacted
    assert "[REDACTED]" in redacted
