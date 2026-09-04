"""Inspect a bounded portion of the public OCF SEVIRI RSS Zarr store."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud2watt.data.probes import ProbeError, ProbeReport, summarize_zarr_metadata, write_report

DEFAULT_STORE = (
    "https://storage.googleapis.com/public-datasets-eumetsat-solar-forecasting/"
    "satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-url", default=DEFAULT_STORE)
    parser.add_argument(
        "--chunk-key",
        default="data/0.3.22.0",
        help="One bounded Zarr v2 chunk to stream; use an empty string for metadata only.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/probes/satellite.json"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=Path("outputs/probes/satellite_preview.png"),
    )
    parser.add_argument("--latitude", type=float, default=52.5)
    parser.add_argument("--longitude", type=float, default=-1.5)
    parser.add_argument("--time-index", type=int, default=144)
    parser.add_argument("--channel-index", type=int, default=7)
    return parser.parse_args()


def fetch_json(url: str, timeout: float) -> tuple[dict[str, Any], dict[str, str]]:
    request = Request(url, headers={"User-Agent": "cloud2watt-data-probe/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        headers = {
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
        }
    return payload, headers


def hash_remote_object(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "cloud2watt-data-probe/0.1"})
    digest = hashlib.sha256()
    size = 0
    with urlopen(request, timeout=timeout) as response:
        while block := response.read(1024 * 1024):
            size += len(block)
            digest.update(block)
        content_type = response.headers.get("Content-Type", "")
    if size == 0:
        raise ProbeError(f"Remote object was empty: {url}")
    return {
        "url": url,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "content_type": content_type,
    }


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "cloud2watt-data-probe/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def decode_zarr_chunk(
    store_url: str,
    metadata: dict[str, Any],
    array_name: str,
    chunk_key: str,
    timeout: float,
) -> Any:
    try:
        import numpy as np
        from numcodecs.registry import get_codec, register_codec
        from ocf_blosc2 import Blosc2
    except ImportError as exc:
        raise ProbeError('Install preview dependencies with: pip install -e ".[data]"') from exc

    register_codec(Blosc2)
    array_metadata = metadata["metadata"][f"{array_name}/.zarray"]
    encoded = fetch_bytes(f"{store_url}/{array_name}/{chunk_key}", timeout)
    decoded = get_codec(array_metadata["compressor"]).decode(encoded)
    return np.frombuffer(decoded, dtype=np.dtype(array_metadata["dtype"]))


def create_preview(
    store_url: str,
    metadata: dict[str, Any],
    *,
    latitude: float,
    longitude: float,
    time_index: int,
    channel_index: int,
    timeout: float,
    output: Path,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise ProbeError('Install preview dependencies with: pip install -e ".[data]"') from exc

    array = metadata["metadata"]["data/.zarray"]
    shape = array["shape"]
    chunks = array["chunks"]
    if not (0 <= time_index < shape[0] and 0 <= channel_index < shape[3]):
        raise ProbeError("time-index or channel-index is outside the data array")

    x = decode_zarr_chunk(store_url, metadata, "x_geostationary", "0", timeout)
    y = decode_zarr_chunk(store_url, metadata, "y_geostationary", "0", timeout)
    crs = CRS.from_proj4(
        "+proj=geos +lon_0=9.5 +h=35785831 +a=6378169 "
        "+rf=295.488065897014 +sweep=x +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    target_x, target_y = transformer.transform(longitude, latitude)
    x_index = int(np.nanargmin(np.abs(x - target_x)))
    y_index = int(np.nanargmin(np.abs(y - target_y)))

    chunk_indices = [
        time_index // chunks[0],
        y_index // chunks[1],
        x_index // chunks[2],
        channel_index // chunks[3],
    ]
    chunk_key = ".".join(str(index) for index in chunk_indices)
    values = decode_zarr_chunk(store_url, metadata, "data", chunk_key, timeout)
    actual_chunk_shape = [
        min(chunk_size, dimension_size - chunk_index * chunk_size)
        for chunk_size, dimension_size, chunk_index in zip(
            chunks, shape, chunk_indices, strict=True
        )
    ]
    expected = int(np.prod(actual_chunk_shape))
    if values.size != expected:
        raise ProbeError(f"Decoded chunk has {values.size} values; expected {expected}")
    values = values.reshape(actual_chunk_shape)

    channel_names = decode_zarr_chunk(store_url, metadata, "variable", "0", timeout)
    time_chunk_size = metadata["metadata"]["time/.zarray"]["chunks"][0]
    time_chunk_index, local_time_coordinate = divmod(time_index, time_chunk_size)
    time_values = decode_zarr_chunk(store_url, metadata, "time", str(time_chunk_index), timeout)
    channel_name = str(channel_names[channel_index])
    timestamp = str(np.datetime64(int(time_values[local_time_coordinate]), "ns"))

    local_time = time_index % chunks[0]
    local_y = y_index % chunks[1]
    local_x = x_index % chunks[2]
    local_channel = channel_index % chunks[3]
    image = values[local_time, :, :, local_channel]

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    plot = axis.imshow(image, cmap="gray")
    axis.scatter([local_x], [local_y], marker="+", s=180, linewidths=2, color="red")
    axis.set_title(f"SEVIRI {channel_name} near ({latitude:.3f}, {longitude:.3f})\n{timestamp} UTC")
    axis.set_xlabel("x within Zarr chunk")
    axis.set_ylabel("y within Zarr chunk")
    figure.colorbar(plot, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)

    return {
        "output": str(output),
        "latitude": latitude,
        "longitude": longitude,
        "projected_x_m": target_x,
        "projected_y_m": target_y,
        "x_index": x_index,
        "y_index": y_index,
        "time_index": time_index,
        "channel_index": channel_index,
        "channel_name": channel_name,
        "timestamp_utc": timestamp,
        "chunk_key": f"data/{chunk_key}",
        "finite_fraction": float(np.isfinite(image).mean()),
        "minimum": float(np.nanmin(image)),
        "maximum": float(np.nanmax(image)),
    }


def run(args: argparse.Namespace) -> ProbeReport:
    store_url = args.store_url.rstrip("/")
    metadata_url = f"{store_url}/.zmetadata"
    payload, headers = fetch_json(metadata_url, args.timeout)
    summary = summarize_zarr_metadata(payload)

    details: dict[str, Any] = {
        "store_url": store_url,
        "metadata_url": metadata_url,
        "metadata_headers": headers,
        "array": summary,
        "chunk_checked": False,
    }
    if args.chunk_key:
        chunk_url = f"{store_url}/{args.chunk_key.lstrip('/')}"
        details["chunk"] = hash_remote_object(chunk_url, args.timeout)
        details["chunk_checked"] = True

    if args.preview:
        details["preview"] = create_preview(
            store_url,
            payload,
            latitude=args.latitude,
            longitude=args.longitude,
            time_index=args.time_index,
            channel_index=args.channel_index,
            timeout=args.timeout,
            output=args.preview_output,
        )

    return ProbeReport.create("ocf_seviri_rss", "ok", details)


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (HTTPError, URLError, TimeoutError, ProbeError, json.JSONDecodeError) as exc:
        report = ProbeReport.create("ocf_seviri_rss", "failed", {"error": str(exc)})
        write_report(args.output, report)
        print(f"Satellite probe failed: {exc}", file=sys.stderr)
        return 1

    write_report(args.output, report)
    print(f"Satellite probe succeeded; report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
