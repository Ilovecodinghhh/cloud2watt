"""Lossless sparse spatial tiles shared across overlapping site crops.

Migration preserves historical pixel indices. It does not correct geolocation
or change the meaning of labels. A new geolocation requires a new source build.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import zarr

SCHEMA = "cloud2watt-shared-tiles-v1"


def intersections(y, x, height, width, tile_size):
    """Yield tile origins and exact intersection bounds, including cross-tile halo."""
    if min(y, x) < 0 or min(height, width, tile_size) < 1:
        raise ValueError("invalid crop bounds")
    for ty in range(y // tile_size * tile_size, y + height, tile_size):
        for tx in range(x // tile_size * tile_size, x + width, tile_size):
            yield ty, tx, max(y, ty), min(y + height, ty + tile_size), \
                max(x, tx), min(x + width, tx + tile_size)


def migrate_site_frames(source, output, sites, *, tile_size=128, time_chunk=12,
                        frame_indices=None):
    """Stream old crops into their union; reject inconsistent overlap, never overwrite.

    Pixels outside the crop union remain NaN and have coverage=False. Missing
    frames retain NaN. This bounded migration cannot invent additional halo.
    """
    if tile_size < 1 or time_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    old = zarr.open_group(source, mode="r")
    frames = old["frames"]
    count, times, channels, height, width = frames.shape
    if len(sites) != count or sites.ss_id.astype(str).tolist() != old.attrs["site_ids"]:
        raise ValueError("site order must match source frame order")
    indices = np.arange(times) if frame_indices is None else np.asarray(frame_indices, dtype=int)
    if (not len(indices) or (indices < 0).any() or (indices >= times).any()
            or (np.diff(indices) <= 0).any()):
        raise ValueError("frame indices must be increasing unique stored slots")
    origins = [(int(s.satellite_y_index) - height // 2,
                int(s.satellite_x_index) - width // 2) for s in sites.itertuples()]
    pieces = [list(intersections(y, x, height, width, tile_size)) for y, x in origins]
    tiles = sorted({(p[0], p[1]) for crop in pieces for p in crop})
    output.mkdir(parents=True, exist_ok=False)
    root = zarr.open_group(output, mode="w")
    root.attrs.update({"schema": SCHEMA, "complete": False, "tile_size": tile_size,
                       "time_chunk": time_chunk, "site_origins": origins,
                       "site_ids": old.attrs["site_ids"], "crop_shape": [height, width],
                       "channel_names": old.attrs["channel_names"],
                       "source": str(source), "geometry": "historical_indices_preserved",
                       "tiles": tiles})
    root.create_array("time_ns", data=np.asarray(old["time_ns"][:])[indices])
    root.create_array("source_frame_indices", data=indices)
    arrays = {}
    for ty, tx in tiles:
        arrays[ty, tx] = root.create_array(
            f"tiles/{ty}_{tx}", shape=(len(indices), channels, tile_size, tile_size),
            chunks=(time_chunk, channels, tile_size, tile_size), dtype=frames.dtype,
            fill_value=np.nan)
        coverage = np.zeros((tile_size, tile_size), dtype=bool)
        for crop in pieces:
            for py, px, y0, y1, x0, x1 in crop:
                if (py, px) == (ty, tx):
                    coverage[y0-ty:y1-ty, x0-tx:x1-tx] = True
        root.create_array(f"coverage/{ty}_{tx}", data=coverage)
    for start in range(0, len(indices), time_chunk):
        stop = min(start + time_chunk, len(indices))
        blocks = {key: np.full((stop-start, channels, tile_size, tile_size), np.nan,
                               dtype=frames.dtype) for key in tiles}
        written = {key: np.zeros((tile_size, tile_size), dtype=bool) for key in tiles}
        for site, ((y, x), crop) in enumerate(zip(origins, pieces, strict=True)):
            values = np.asarray(frames[site, indices[start:stop]])
            for ty, tx, y0, y1, x0, x1 in crop:
                dest = blocks[ty, tx][:, :, y0-ty:y1-ty, x0-tx:x1-tx]
                incoming = values[:, :, y0-y:y1-y, x0-x:x1-x]
                overlap = written[ty, tx][y0-ty:y1-ty, x0-tx:x1-tx]
                if overlap.any() and not np.array_equal(
                        dest[:, :, overlap], incoming[:, :, overlap], equal_nan=True):
                    raise ValueError("overlapping source crops disagree; migration incomplete")
                dest[:] = incoming
                overlap[:] = True
        for key, block in blocks.items():
            arrays[key][start:stop] = block
    root.attrs["complete"] = True
    return {"schema": SCHEMA, "sites": count, "frames": len(indices),
            "tiles": len(tiles), "tile_size": tile_size,
            "logical_source_pixels": count * height * width,
            "stored_tile_pixels": len(tiles) * tile_size**2,
            "covered_pixels": sum(int(root[f"coverage/{y}_{x}"][:].sum()) for y, x in tiles)}


class SharedFrames:
    """Bounded per-worker decoded-block cache with the legacy frame indexing API."""

    def __init__(self, root, *, cache_blocks=16):
        if root.attrs.get("schema") != SCHEMA or not root.attrs.get("complete"):
            raise ValueError("incomplete or unsupported shared store")
        if cache_blocks < 0:
            raise ValueError("cache_blocks must be nonnegative")
        self.root, self.cache_blocks = root, cache_blocks
        self.cache = OrderedDict()
        self.origins = root.attrs["site_origins"]
        self.tile_size, self.time_chunk = root.attrs["tile_size"], root.attrs["time_chunk"]
        self.arrays = {(y, x): root[f"tiles/{y}_{x}"] for y, x in root.attrs["tiles"]}
        self.coverage = {(y, x): root[f"coverage/{y}_{x}"][:] for y, x in self.arrays}
        self.shape = (len(self.origins), root["time_ns"].shape[0],
                      len(root.attrs["channel_names"]), *root.attrs["crop_shape"])
        self.dtype = next(iter(self.arrays.values())).dtype

    def read(self, site, indices, *, halo=0):
        if not 0 <= site < self.shape[0] or halo < 0:
            raise ValueError("invalid site or halo")
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if (indices < 0).any() or (indices >= self.shape[1]).any():
            raise IndexError("frame index outside stored slots")
        y, x = self.origins[site]
        y, x = y-halo, x-halo
        height, width = self.shape[3]+2*halo, self.shape[4]+2*halo
        output = np.empty((len(indices), self.shape[2], height, width), dtype=self.dtype)
        for ty, tx, y0, y1, x0, x1 in intersections(y, x, height, width, self.tile_size):
            if ((ty, tx) not in self.coverage or not np.all(
                    self.coverage[ty, tx][y0-ty:y1-ty, x0-tx:x1-tx])):
                raise ValueError("requested halo outside stored coverage; rebuild from source")
            for group in np.unique(indices // self.time_chunk):
                key = (ty, tx, int(group))
                block = self.cache.pop(key, None)
                if block is None:
                    start = int(group) * self.time_chunk
                    block = np.asarray(self.arrays[ty, tx][start:start+self.time_chunk])
                if self.cache_blocks:
                    self.cache[key] = block
                    while len(self.cache) > self.cache_blocks:
                        self.cache.popitem(last=False)
                take = indices // self.time_chunk == group
                output[take, :, y0-y:y1-y, x0-x:x1-x] = block[
                    indices[take] % self.time_chunk, :, y0-ty:y1-ty, x0-tx:x1-tx]
        return output

    def __getitem__(self, key):
        site, indices = key
        scalar = np.isscalar(indices)
        if isinstance(indices, slice):
            indices = np.arange(self.shape[1])[indices]
        result = self.read(int(site), indices)
        return result[0] if scalar else result


def open_satellite_frames(path):
    root = zarr.open_group(path, mode="r")
    if root.attrs.get("schema") == SCHEMA:
        return SharedFrames(root)
    return root["frames"]


def store_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
