"""Bounded reader for the public OCF SEVIRI RSS Zarr v2 archive."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from numcodecs.registry import get_codec, register_codec
from ocf_blosc2 import Blosc2
from pyproj import CRS, Transformer

from cloud2watt.data.spatial import nearest_index

SEVIRI_PROJ4 = (
    "+proj=geos +lon_0=9.5 +h=35785831 +a=6378169 +rf=295.488065897014 +sweep=x +units=m +no_defs"
)


@dataclass(frozen=True)
class SitePixel:
    """One PV site mapped into the SEVIRI grid."""

    site_id: str
    x_index: int
    y_index: int
    projected_x_m: float
    projected_y_m: float


def reshape_zarr_chunk(
    values: np.ndarray, spec: dict[str, Any], chunk_indices: list[int]
) -> np.ndarray:
    """Trim either compact or full padded edge chunks to their logical shape."""
    actual_shape = [
        min(chunk, size - index * chunk)
        for chunk, size, index in zip(spec["chunks"], spec["shape"], chunk_indices, strict=True)
    ]
    actual_size = int(np.prod(actual_shape))
    full_size = int(np.prod(spec["chunks"]))
    if values.size == actual_size:
        return values.reshape(actual_shape, order=spec.get("order", "C"))
    if values.size == full_size:
        padded = values.reshape(spec["chunks"], order=spec.get("order", "C"))
        return padded[tuple(slice(0, size) for size in actual_shape)]
    raise ValueError(
        f"decoded chunk has {values.size} values; expected {actual_size} or {full_size}"
    )


class SeviriStore:
    """Read only selected chunks from an HTTP-hosted consolidated Zarr v2 store."""

    def __init__(
        self,
        store_url: str,
        *,
        cache_dir: Path,
        timeout: float = 60.0,
        decoded_cache_size: int = 32,
        max_attempts: int = 4,
    ) -> None:
        self.store_url = store_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.decoded_cache_size = decoded_cache_size
        self.max_attempts = max_attempts
        self._decoded: OrderedDict[str, np.ndarray] = OrderedDict()
        self.downloaded_bytes = 0
        self.cache_hits = 0
        self.network_fetches = 0
        self.retry_count = 0
        register_codec(Blosc2)
        payload = self._fetch_bytes(".zmetadata")
        self.metadata_sha256 = hashlib.sha256(payload).hexdigest()
        self.metadata: dict[str, Any] = json.loads(payload)["metadata"]
        self.data_spec = self.metadata["data/.zarray"]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir.joinpath(*key.split("/"))

    def _fetch_bytes(self, key: str) -> bytes:
        cache_path = self._cache_path(key)
        if cache_path.exists():
            self.cache_hits += 1
            return cache_path.read_bytes()
        request = Request(
            f"{self.store_url}/{key}", headers={"User-Agent": "cloud2watt-pairing/0.1"}
        )
        payload = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                break
            except (HTTPError, URLError, TimeoutError):
                if attempt == self.max_attempts:
                    raise
                self.retry_count += 1
                time.sleep(2 ** (attempt - 1))
        if payload is None:
            raise RuntimeError(f"failed to fetch {key}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
        self.downloaded_bytes += len(payload)
        self.network_fetches += 1
        return payload

    def prefetch(self, keys: list[str], *, workers: int = 4) -> None:
        """Download unique encoded chunks concurrently into the local cache."""
        unique_keys = sorted(set(keys))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(self._fetch_bytes, unique_keys))

    def decode_chunk(self, array_name: str, chunk_key: str) -> np.ndarray:
        """Decode one chunk and retain a small least-recently-used memory cache."""
        object_key = f"{array_name}/{chunk_key}"
        if object_key in self._decoded:
            result = self._decoded.pop(object_key)
            self._decoded[object_key] = result
            return result
        spec = self.metadata[f"{array_name}/.zarray"]
        encoded = self._fetch_bytes(object_key)
        decoded = get_codec(spec["compressor"]).decode(encoded)
        result = np.frombuffer(decoded, dtype=np.dtype(spec["dtype"]))
        chunk_indices = [int(value) for value in chunk_key.split(".")]
        result = reshape_zarr_chunk(result, spec, chunk_indices)
        self._decoded[object_key] = result
        while len(self._decoded) > self.decoded_cache_size:
            self._decoded.popitem(last=False)
        return result

    def coordinate(self, name: str) -> np.ndarray:
        """Decode a one-dimensional coordinate array."""
        return self.decode_chunk(name, "0")

    def timestamps(self) -> pd.DatetimeIndex:
        """Decode every small time-coordinate chunk as UTC."""
        spec = self.metadata["time/.zarray"]
        chunk_count = (spec["shape"][0] + spec["chunks"][0] - 1) // spec["chunks"][0]
        values = np.concatenate([self.decode_chunk("time", str(i)) for i in range(chunk_count)])
        return pd.DatetimeIndex(pd.to_datetime(values, unit="ns", utc=True))

    def channels(self) -> list[str]:
        return [str(value) for value in self.coordinate("variable")]

    def map_sites(self, sites: pd.DataFrame) -> list[SitePixel]:
        """Project site coordinates and map them to nearest SEVIRI pixels."""
        required = {"ss_id", "latitude_rounded", "longitude_rounded"}
        missing = required.difference(sites.columns)
        if missing:
            raise ValueError(f"missing site columns: {sorted(missing)}")
        x = self.coordinate("x_geostationary")
        y = self.coordinate("y_geostationary")
        transformer = Transformer.from_crs(
            "EPSG:4326", CRS.from_proj4(SEVIRI_PROJ4), always_xy=True
        )
        result = []
        for row in sites.itertuples(index=False):
            projected_x, projected_y = transformer.transform(
                row.longitude_rounded, row.latitude_rounded
            )
            result.append(
                SitePixel(
                    site_id=str(row.ss_id),
                    x_index=nearest_index(x, projected_x),
                    y_index=nearest_index(y, projected_y),
                    projected_x_m=float(projected_x),
                    projected_y_m=float(projected_y),
                )
            )
        return result

    def required_data_keys(
        self,
        time_indices: list[int],
        sites: list[SitePixel],
        *,
        height: int,
        width: int,
    ) -> list[str]:
        """Return the encoded data chunks required for all crops."""
        chunks = self.data_spec["chunks"]
        keys = []
        for time_chunk in sorted({index // chunks[0] for index in time_indices}):
            for site in sites:
                row_start = site.y_index - height // 2
                column_start = site.x_index - width // 2
                self._validate_crop(row_start, column_start, height, width)
                y_chunk_stop = (row_start + height - 1) // chunks[1] + 1
                for y_chunk in range(row_start // chunks[1], y_chunk_stop):
                    for x_chunk in range(
                        column_start // chunks[2],
                        (column_start + width - 1) // chunks[2] + 1,
                    ):
                        keys.append(f"data/{time_chunk}.{y_chunk}.{x_chunk}.0")
        return sorted(set(keys))

    def read_crop(
        self,
        *,
        time_index: int,
        site: SitePixel,
        channel_indices: list[int],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Assemble one exact crop, including boundaries between storage chunks."""
        chunks = self.data_spec["chunks"]
        row_start = site.y_index - height // 2
        column_start = site.x_index - width // 2
        self._validate_crop(row_start, column_start, height, width)
        output = np.empty((len(channel_indices), height, width), dtype=np.float16)
        time_chunk, local_time = divmod(time_index, chunks[0])
        for y_chunk in range(row_start // chunks[1], (row_start + height - 1) // chunks[1] + 1):
            source_y_start = y_chunk * chunks[1]
            y0 = max(row_start, source_y_start)
            y1 = min(row_start + height, source_y_start + chunks[1])
            for x_chunk in range(
                column_start // chunks[2], (column_start + width - 1) // chunks[2] + 1
            ):
                source_x_start = x_chunk * chunks[2]
                x0 = max(column_start, source_x_start)
                x1 = min(column_start + width, source_x_start + chunks[2])
                chunk = self.decode_chunk("data", f"{time_chunk}.{y_chunk}.{x_chunk}.0")
                source = chunk[
                    local_time,
                    y0 - source_y_start : y1 - source_y_start,
                    x0 - source_x_start : x1 - source_x_start,
                    :,
                ]
                source = source[..., channel_indices]
                output[
                    :, y0 - row_start : y1 - row_start, x0 - column_start : x1 - column_start
                ] = np.moveaxis(source, -1, 0)
        return output

    def _validate_crop(self, row_start: int, column_start: int, height: int, width: int) -> None:
        shape = self.data_spec["shape"]
        if (
            row_start < 0
            or column_start < 0
            or row_start + height > shape[1]
            or column_start + width > shape[2]
        ):
            raise ValueError("site crop extends beyond the SEVIRI domain")
