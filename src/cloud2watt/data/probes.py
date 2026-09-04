"""Pure helpers shared by the external data access probes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ProbeError(RuntimeError):
    """Raised when a remote data source does not satisfy the expected contract."""


@dataclass(frozen=True)
class RemoteFile:
    """Minimal remote-file metadata used to select a bounded sample."""

    path: str
    size: int | None


@dataclass(frozen=True)
class ProbeReport:
    """Serializable result of one data-access probe."""

    source: str
    status: str
    checked_at_utc: str
    details: Mapping[str, Any]

    @classmethod
    def create(cls, source: str, status: str, details: Mapping[str, Any]) -> ProbeReport:
        return cls(
            source=source,
            status=status,
            checked_at_utc=datetime.now(UTC).isoformat(),
            details=details,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_smallest_file(
    files: Iterable[RemoteFile], *, prefix: str, suffix: str
) -> RemoteFile:
    """Choose the smallest sized file within a constrained dataset subtree."""
    candidates = [
        file
        for file in files
        if file.path.startswith(prefix) and file.path.endswith(suffix) and file.size is not None
    ]
    if not candidates:
        raise ProbeError(f"No sized file matched prefix={prefix!r}, suffix={suffix!r}")
    return min(candidates, key=lambda file: (file.size or 0, file.path))


def summarize_zarr_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the stable storage contract from consolidated Zarr v2 metadata."""
    try:
        metadata = payload["metadata"]
        group = metadata[".zgroup"]
        array = metadata["data/.zarray"]
        attributes = metadata["data/.zattrs"]
    except (KeyError, TypeError) as exc:
        raise ProbeError(f"Unexpected consolidated Zarr metadata: missing {exc}") from exc

    dimensions = attributes.get("_ARRAY_DIMENSIONS")
    if group.get("zarr_format") != 2 or not isinstance(dimensions, list):
        raise ProbeError("Expected a Zarr v2 data array with named dimensions")

    shape = array.get("shape")
    chunks = array.get("chunks")
    if not isinstance(shape, list) or not isinstance(chunks, list) or len(shape) != len(dimensions):
        raise ProbeError("Zarr data shape, chunks, and dimensions are inconsistent")

    compressor = array.get("compressor") or {}
    return {
        "zarr_format": group["zarr_format"],
        "dimensions": dimensions,
        "shape": shape,
        "chunks": chunks,
        "dtype": array.get("dtype"),
        "compressor": compressor,
        "platform_name": attributes.get("platform_name"),
        "sensor": attributes.get("sensor"),
        "resolution_m": attributes.get("resolution"),
        "start_time": attributes.get("start_time"),
        "end_time": attributes.get("end_time"),
    }


def redact_secrets(message: str) -> str:
    """Remove common bearer-token forms before persisting an error message."""
    redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[REDACTED]", message)
    redacted = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "hf_[REDACTED]", redacted)
    return redacted


def write_report(path: Path, report: ProbeReport) -> None:
    """Write a deterministic UTF-8 JSON report without creating a BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
