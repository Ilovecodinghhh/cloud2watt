"""Bounded metadata/geometry preflight for route B; never fetch PV test labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from pyproj import Geod

from cloud2watt.data.seviri import SeviriStore
from cloud2watt.run_state import atomic_json, file_hash, json_hash

WINDOWS = [
    {"id": "winter_fit", "role": "development", "start": "2021-01-01T00:00Z",
     "end": "2021-01-29T00:00Z"},
    {"id": "spring", "role": "development", "start": "2021-04-01T00:00Z",
     "end": "2021-05-07T00:00Z"},
    {"id": "summer", "role": "development", "start": "2021-07-01T00:00Z",
     "end": "2021-08-06T00:00Z"},
    {"id": "autumn", "role": "development", "start": "2021-10-01T00:00Z",
     "end": "2021-11-06T00:00Z"},
    {"id": "final_winter", "role": "locked_final", "start": "2022-01-01T00:00Z",
     "end": "2022-02-26T00:00Z"},
]


def select_regions(metadata, excluded, *, count=4, sites_per_region=10):
    """Choose dense compact coordinate cells, separated by >=150 km; no power values."""
    frame = metadata.copy()
    begin = pd.to_datetime(frame.start_datetime_GMT, utc=True)
    end = pd.to_datetime(frame.end_datetime_GMT, utc=True)
    frame = frame.loc[(begin <= "2021-01-01") & (end >= "2022-02-26") & (frame.kWp > 0)]
    frame = frame.dropna(subset=["latitude_rounded", "longitude_rounded"]).copy()
    frame["lat_cell"] = np.floor(frame.latitude_rounded * 4).astype(int)
    frame["lon_cell"] = np.floor(frame.longitude_rounded * 4).astype(int)
    counts = frame.groupby(["lat_cell", "lon_cell"]).size().reset_index(name="count")
    counts = counts.sort_values(["count", "lat_cell", "lon_cell"], ascending=[False, True, True])
    geod, centres, selected = Geod(ellps="WGS84"), [], []
    for cell in counts.itertuples(index=False):
        if cell.count < sites_per_region:
            continue
        group = frame.loc[(frame.lat_cell == cell.lat_cell) & (frame.lon_cell == cell.lon_cell)]
        lat, lon = (cell.lat_cell + .5) / 4, (cell.lon_cell + .5) / 4
        if any(geod.inv(lon, lat, x, y)[2] < 150_000 for x, y in centres):
            continue
        final = len(selected) == count - 1
        if final and (group.ss_id.astype(str).isin(excluded).any()):
            continue
        group = group.assign(rank=group.ss_id.map(
            lambda value: hashlib.sha256(f"2026:{value}".encode()).hexdigest()))
        group = group.sort_values(["rank", "ss_id"]).head(sites_per_region)
        centres.append((lon, lat))
        selected.append({"region_id": f"r{len(selected)}", "centre": [lon, lat],
                         "coordinate_cell": [int(cell.lat_cell), int(cell.lon_cell)],
                         "role": "locked_geography" if final else "development",
                         "site_ids": group.ss_id.astype(str).tolist()})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("insufficient metadata-only separated geographic groups")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(Path("outputs/probes/uk_pv.json").read_text("utf-8"))["details"]
    old_path = Path("data/processed/paired-30d-corrected-v2/manifest.json")
    old = json.loads(old_path.read_text("utf-8"))
    atomic_json(args.output / "preregistration.json", {
        "stage": "B-data-preflight", "test_access": "forbidden", "windows": WINDOWS,
        "region_selection": "metadata-only 0.25 degree cells; dense-first; centres >=150km",
        "max_satellite_sample_objects": 2, "max_sample_object_bytes": 16 * 1024**2,
        "source": old["source"], "label_contract": "c2w-sampled-power-v1",
        "pv_eligibility": "metadata plus site IDs present in January 2021 five-minute file",
    })
    metadata_path = Path(previous["metadata"]["path"])
    metadata = pd.read_csv(metadata_path)
    fit_file = Path(hf_hub_download("openclimatefix/uk_pv",
        "5_minutely/year=2021/month=01/data.parquet", repo_type="dataset",
        revision=previous["revision"], cache_dir=args.output / "pv_source"))
    # Site availability only; no generation_Wh or final-period labels are materialized.
    fit_ids = set(pq.read_table(fit_file, columns=["ss_id"]).column("ss_id").to_pylist())
    metadata = metadata.loc[metadata.ss_id.isin(fit_ids)]
    regions = select_regions(metadata, set(old["site_ids"]))
    ids = [site for region in regions for site in region["site_ids"]]
    sites = metadata.loc[metadata.ss_id.astype(str).isin(ids)].copy()
    sites.to_csv(args.output / "candidate_sites.csv", index=False, encoding="utf-8")
    info = HfApi().dataset_info("openclimatefix/uk_pv", revision=previous["revision"],
                              files_metadata=True, timeout=30)
    files = {item.rfilename: item.size for item in info.siblings}
    required_pv = {}
    stores, availability = {}, []
    for year in (2021, 2022):
        url = old["source"]["satellite"].replace("2020_nonhrv", f"{year}_nonhrv")
        cache = args.source_cache or args.output / "source_metadata"
        stores[year] = SeviriStore(url, cache_dir=cache / str(year),
                                   timeout=20, max_attempts=2)
        times = stores[year].timestamps()
        for window in WINDOWS:
            if int(window["start"][:4]) != year:
                continue
            desired = pd.date_range(window["start"], window["end"], freq="15min",
                                    inclusive="left")
            availability.append({"id": window["id"], "expected_slots": len(desired),
                                 "present_slots": int((times.get_indexer(desired) >= 0).sum()),
                                 "coordinate_only": True})
            for month in pd.period_range(desired[0].tz_localize(None),
                                         desired[-1].tz_localize(None), freq="M"):
                key = f"5_minutely/year={month.year}/month={month.month:02d}/data.parquet"
                if key not in files:
                    raise ValueError(f"missing source partition: {key}")
                required_pv[key] = files[key]
    client = stores[2021]
    spatial = []
    for region in regions:
        chosen = sites.loc[sites.ss_id.astype(str).isin(region["site_ids"])]
        pixels = client.map_sites(chosen)
        source_keys = client.required_data_keys([0], pixels, height=128, width=128)
        spatial.append({"region_id": region["region_id"], "source_chunks_per_time_chunk":
                        len(source_keys), "source_keys": source_keys,
                        "pixels": [[p.x_index, p.y_index] for p in pixels]})
    # At most two actual image objects, both from an early development region/date.
    samples = []
    first_time = client.timestamps().get_indexer(pd.DatetimeIndex(["2021-01-01T12:00Z"]))[0]
    if first_time < 0:
        raise ValueError("pre-registered development probe frame is missing")
    dev_sites = sites.loc[sites.ss_id.astype(str).isin(regions[0]["site_ids"])]
    sample_keys = client.required_data_keys([int(first_time)], client.map_sites(dev_sites),
                                           height=128, width=128)[:2]
    for key in sample_keys:
        # HEAD first prevents unexpectedly large objects from being fetched for this probe.
        from urllib.request import Request, urlopen
        with urlopen(Request(f"{client.store_url}/{key}", method="HEAD"), timeout=20) as response:
            size = int(response.headers["Content-Length"])
        if size > 16 * 1024**2:
            raise ValueError("preflight image object exceeds bounded probe budget")
        payload = client._fetch_bytes(key)
        samples.append({"key": key, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()})
    folds = []
    for index, window in enumerate(WINDOWS[1:4]):
        start = pd.Timestamp(window["start"])
        folds.append({"fold": index, "fit_windows": [w["id"] for w in WINDOWS[:index+1]],
                      "inner_start": start.isoformat(),
                      "inner_end": (start + pd.Timedelta(days=7)).isoformat(),
                      "outer_start": (start + pd.Timedelta(days=7, hours=4)).isoformat(),
                      "outer_end": window["end"], "embargo_hours": 4,
                      "require_complete_input_and_target_support": True})
    disk = shutil.disk_usage("D:/")
    report = {"status": "preflight_only_not_training_ready", "pv_revision": info.sha,
              "metadata_sha256": file_hash(metadata_path), "regions": regions,
              "fit_id_source_sha256": file_hash(fit_file), "five_minute_site_count": len(fit_ids),
              "windows": WINDOWS, "folds": folds, "satellite_coordinate_coverage": availability,
              "satellite_sources": {str(y): {"url": c.store_url,
                "metadata_sha256": c.metadata_sha256, "data_spec": c.data_spec}
                for y, c in stores.items()},
              "spatial": spatial, "sample_objects": samples,
              "required_pv_partitions": required_pv, "pv_download_bytes": sum(required_pv.values()),
              "disk_free_bytes": disk.free, "disk_reserve_bytes": int(.2 * disk.total),
              "bytes_above_reserve": max(0, disk.free - int(.2 * disk.total)),
              "pv_labels_read": False, "test_labels_read": False,
              "remaining_gates": ["bounded streaming tile cache", "actual development QC",
                                  "source crop/geometry validation", "isolated final-test files",
                                  "resource estimate by actual source chunk counts"],
              "new_source_bytes": sum(c.downloaded_bytes for c in stores.values())}
    report["preflight_sha256"] = json_hash(report)
    atomic_json(args.output / "preflight.json", report)
    print(json.dumps({"regions": regions, "coverage": availability,
                      "pv_download_bytes": sum(required_pv.values()),
                      "new_source_bytes": report["new_source_bytes"]}))


if __name__ == "__main__":
    main()
