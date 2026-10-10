"""Single-writer, resumable route B development download and partition builder."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import google_crc32c
import pandas as pd
import psutil
from filelock import FileLock
from huggingface_hub import hf_hub_download

from cloud2watt.data.bounded_source import BoundedSeviriStore
from cloud2watt.data.seviri import SeviriStore
from cloud2watt.data.shared_tiles import store_bytes
from cloud2watt.run_state import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "outputs/project-v2/route-b-download"
DEST = ROOT / "data/processed/route-b-development-v1"
CACHE = ROOT / "outputs/project-v2/source-cache-2021"
QUARANTINE = ROOT / "data/quarantine/route-b-final-v1"
FREEZE = ROOT / "configs/project-v2/route-b-freeze.json"
PREFLIGHT = ROOT / "outputs/project-v2/b-preflight-five-minute-20261001/preflight.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CheckedDevelopmentStore(BoundedSeviriStore):
    """Reject truncated development Blosc blocks before native decoding."""

    def _fetch_bytes(self, key):
        if not key.startswith("data/"):
            return super()._fetch_bytes(key)
        ledger_path = self.root / "integrity-failures.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else {}
        while True:
            if ledger.get(key, 0) >= 3:
                raise RuntimeError(f"source integrity retry limit: {key}")
            payload = super()._fetch_bytes(key)
            if len(payload) >= 16 and int.from_bytes(payload[12:16], "little") == len(payload):
                return payload
            ledger[key] = ledger.get(key, 0) + 1
            atomic_json(ledger_path, ledger)
            path = self._cache_path(key)
            target = self.root / "corrupt" / key / f"attempt-{ledger[key]}"
            if not target.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("unsafe corrupt cache path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            path.rename(target)
            print(f"TRUNCATED SOURCE quarantined {key}: {len(payload)} bytes", flush=True)


def load_builder():
    path = ROOT / "scripts/build_route_b_pilot.py"
    spec = importlib.util.spec_from_file_location("route_b_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tasks_for(plan):
    """One UTC issue date per shard; QC halo never crosses seasonal boundaries."""
    tasks = []
    for window in plan["windows"]:
        if window["role"] != "development":
            continue
        first, last = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        for day in pd.date_range(first, last, freq="D", inclusive="left"):
            for region in plan["regions"]:
                if region["role"] != "development":
                    continue
                tasks.append(
                    {
                        "id": f"{window['id']}/{day:%Y%m%d}/{region['region_id']}",
                        "region": region["region_id"],
                        "window": window["id"],
                        "start": max(first, day - pd.Timedelta(hours=3)).isoformat(),
                        "end": min(last, day + pd.Timedelta(days=1, hours=6)).isoformat(),
                        "issue_start": day.isoformat(),
                        "issue_end": (day + pd.Timedelta(days=1)).isoformat(),
                    }
                )
    return tasks


def verify_shard(path):
    """Every persisted artifact, including Zarr chunks, is checked on resume."""
    receipt = read_json(path / "verified.json")
    expected = receipt["files"]
    actual = {
        p.relative_to(path).as_posix()
        for p in path.rglob("*")
        if p.is_file() and p.name != "verified.json"
    }
    if set(expected) != actual:
        raise ValueError(f"partition file inventory changed: {path}")
    for name, digest in expected.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_hash(target) != digest:
            raise ValueError(f"partition hash mismatch: {path}/{name}")
    return receipt


def seal_shard(path, task_id):
    complete = read_json(path / "complete.json")
    for name, digest in {
        **complete["table_hashes"],
        "manifest.json": complete["manifest_sha256"],
        "build_report.json": complete["build_report_sha256"],
    }.items():
        if file_hash(path / name) != digest:
            raise ValueError("builder completion hash mismatch")
    files = {
        p.relative_to(path).as_posix(): file_hash(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "verified.json"
    }
    atomic_json(path / "verified.json", {"task_id": task_id, "files": files})
    return verify_shard(path)


class Budget:
    def __init__(self, config):
        self.limits = config["resource_limits"]
        self.path = STATE / "transfer.json"
        self.state = read_json(self.path) if self.path.exists() else {"pv_reserved_bytes": 0}

    def usage(self):
        paths = [ROOT / "outputs/project-v2", DEST, QUARANTINE]
        paths.extend((ROOT / "data/processed").glob("route-b-pilot-*"))
        return sum(store_bytes(p) for p in paths if p.exists())

    def source_charged(self):
        return (
            read_json(CACHE / "owner.json")["charged_bytes"]
            if (CACHE / "owner.json").exists()
            else 0
        )

    def check(self, growth=0):
        disk = shutil.disk_usage(ROOT)
        if disk.free - growth < disk.total * self.limits["disk_free_fraction_floor"]:
            raise ValueError("disk reserve reached")
        used = self.usage()
        if used + growth > self.limits["new_persistent_bytes_cap"]:
            raise ValueError("aggregate persistent storage budget reached")
        if (
            self.source_charged()
            + self.state["pv_reserved_bytes"]
            + self.state.get("final_reserved_bytes", 0)
            > self.limits["cumulative_source_download_bytes"]
        ):
            raise ValueError("aggregate transfer budget reached")
        return used

    def reserve_pv(self, size):
        charge = size * 4  # conservative allowance for library retries; retained after failure
        if (
            self.source_charged()
            + self.state["pv_reserved_bytes"]
            + self.state.get("final_reserved_bytes", 0)
            + charge
            > self.limits["cumulative_source_download_bytes"]
        ):
            raise ValueError("PV transfer reservation exceeds budget")
        self.check(size * 3)
        self.state["pv_reserved_bytes"] += charge
        atomic_json(self.path, self.state)


def raw_pv(preflight, budget, allowed_names=None):
    """Transfer final PV as opaque bytes; never invoke a table reader here."""
    receipt_path = STATE / "pv-receipts.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else {}
    for name, size in preflight["required_pv_partitions"].items():
        if allowed_names is not None and name not in allowed_names:
            continue
        final = "year=2022/" in name
        cache = QUARANTINE / "pv_source" if final else PREFLIGHT.parent / "pv_source"
        if name in receipts:
            path = Path(receipts[name]["path"])
            if path.exists() and file_hash(path) == receipts[name]["sha256"]:
                continue
            raise ValueError("previously verified raw PV file changed")
        try:
            path = Path(
                hf_hub_download(
                    "openclimatefix/uk_pv",
                    name,
                    repo_type="dataset",
                    revision=preflight["pv_revision"],
                    cache_dir=cache,
                    local_files_only=True,
                )
            )
        except FileNotFoundError:
            attempts = budget.state.setdefault("pv_attempts", {})
            if attempts.get(name, 0) >= 3:
                raise RuntimeError(f"PV retry limit reached for {name}") from None
            attempts[name] = attempts.get(name, 0) + 1
            budget.reserve_pv(size)
            command = [
                sys.executable,
                "-c",
                "import json,sys; from huggingface_hub import hf_hub_download; "
                "print(json.dumps(hf_hub_download('openclimatefix/uk_pv',sys.argv[1],"
                "repo_type='dataset',revision=sys.argv[2],cache_dir=sys.argv[3])))",
                name,
                preflight["pv_revision"],
                str(cache),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "60"},
            )
            if result.returncode:
                raise RuntimeError(
                    f"PV download child failed for {name}: exit {result.returncode}"
                ) from None
            path = Path(json.loads(result.stdout.strip()))
            if not path.resolve().is_relative_to(cache.resolve()):
                raise ValueError("PV download returned a path outside its owned cache") from None
        if path.stat().st_size != size:
            raise ValueError("raw PV transfer size differs from frozen source listing")
        receipts[name] = {
            "path": str(path),
            "bytes": size,
            "sha256": file_hash(path),
            "role": "locked_final_opaque" if final else "development",
            "labels_read_by_download_function": False,
        }
        atomic_json(receipt_path, receipts)
        print(f"PV transfer verified: {name}; locked={final}", flush=True)
    return receipts


def inventory(config, preflight, tasks, client):
    objects = {}
    times = client.timestamps()
    builder = load_builder()
    spatial = {
        r["region_id"]: [k.split(".", 1)[1] for k in r["source_keys"]] for r in preflight["spatial"]
    }
    for task in tasks:
        _, _, _, issue = builder.pilot_bounds(
            preflight, task["region"], task["window"], task["start"], task["end"]
        )
        issue = issue[
            (issue >= pd.Timestamp(task["issue_start"])) & (issue < pd.Timestamp(task["issue_end"]))
        ]
        requested = pd.date_range(issue[0] - pd.Timedelta(minutes=45), issue[-1], freq="15min")
        positions = times.get_indexer(requested)
        chunks = sorted(set(int(i) // 12 for i in positions if i >= 0))
        objects[task["id"]] = [
            f"data/{t}.{tail}" for t in chunks for tail in spatial[task["region"]]
        ]
    final_client = SeviriStore(
        config["satellite_sources"]["2022"]["url"],
        cache_dir=ROOT / "outputs/project-v2/b-preflight-20261001/source_metadata/2022",
    )
    if final_client.metadata_sha256 != config["satellite_sources"]["2022"]["metadata_sha256"]:
        raise ValueError("final satellite metadata drift")
    final = next(w for w in config["windows"] if w["role"] == "locked_final")
    first, last = pd.Timestamp(final["start"]), pd.Timestamp(final["end"])
    requested = pd.date_range(
        first + pd.Timedelta(minutes=15), last - pd.Timedelta(hours=4, minutes=15), freq="15min"
    )
    positions = final_client.timestamps().get_indexer(requested)
    tails = sorted({tail for values in spatial.values() for tail in values})
    final_keys = [
        f"data/{t}.{tail}"
        for t in sorted(set(int(i) // 12 for i in positions if i >= 0))
        for tail in tails
    ]
    unique = sorted({k for keys in objects.values() for k in keys})
    return {
        "development_by_partition": objects,
        "development_unique_keys": unique,
        "final_satellite_keys": final_keys,
        "final_regions": [r["region_id"] for r in config["regions"]],
        "final_status": "pending_exact_storage_budget_check_no_data_objects_downloaded",
        "final_labels_read": False,
        "final_satellite_estimated_bytes_at_pilot_mean": int(len(final_keys) * 211644917 / 176),
        "final_estimate_is_not_HEAD_verified": True,
    }


def final_download(config, budget):
    """Persist exact-generation satellite objects without decoding any payload."""
    sizes_path = STATE / "final-object-sizes.json"
    if not sizes_path.exists():
        return "final_object_sizes_pending"
    sizes = read_json(sizes_path)
    if not sizes.get("complete"):
        return "final_object_sizes_pending"
    if sizes["inventory_sha256"] != file_hash(STATE / "source-inventory.json"):
        raise ValueError("final size inventory identity changed")
    receipt_path = STATE / "final-download-receipts.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else {}
    remaining = sum(v["bytes"] for k, v in sizes["objects"].items() if k not in receipts)
    # Reserve the entire remaining persistent payload before starting; no eviction
    # of final files and no label/image parsing are permitted.
    budget.check(remaining + 32 * 1024**2)
    source = config["satellite_sources"]["2022"]["url"]
    target_root = QUARANTINE / "satellite-2022"
    for key, item in sizes["objects"].items():
        path = (target_root / key).resolve()
        if not path.is_relative_to(target_root.resolve()):
            raise ValueError("invalid final source key")
        if key in receipts:
            if file_hash(path) != receipts[key]["sha256"]:
                raise ValueError("final opaque file changed")
            continue
        if not item.get("generation") or not item.get("hash_header"):
            raise ValueError("final object lacks immutable generation or checksum")
        attempts = budget.state.setdefault("final_attempts", {})
        if attempts.get(key, 0) >= 3:
            raise ValueError("final source retry limit reached")
        # Charge full known response before each request; failures retain charge.
        used_transfer = (
            budget.source_charged()
            + budget.state["pv_reserved_bytes"]
            + budget.state.get("final_reserved_bytes", 0)
        )
        if used_transfer + item["bytes"] > budget.limits["cumulative_source_download_bytes"]:
            raise ValueError("final transfer would exceed aggregate budget")
        attempts[key] = attempts.get(key, 0) + 1
        budget.state["final_reserved_bytes"] = (
            budget.state.get("final_reserved_bytes", 0) + item["bytes"]
        )
        atomic_json(budget.path, budget.state)
        budget.check(item["bytes"] + 1024**2)
        request = Request(f"{source}/{key}?generation={item['generation']}")
        with urlopen(request, timeout=60) as response:
            payload = response.read(min(item["bytes"], budget.limits["source_object_bytes"]) + 1)
        if len(payload) != item["bytes"]:
            raise ValueError("final object byte count mismatch")
        expected = next(
            (
                v.split("=", 1)[1]
                for v in item["hash_header"].split(",")
                if v.strip().startswith("crc32c=")
            ),
            None,
        )
        checksum = base64.b64encode(google_crc32c.value(payload).to_bytes(4, "big")).decode()
        if expected is None or checksum != expected:
            raise ValueError("final source CRC32C mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(payload)
        partial.replace(path)
        receipts[key] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "generation": item["generation"],
            "crc32c": checksum,
        }
        atomic_json(receipt_path, receipts)
        if len(receipts) % 64 == 0:
            print(f"FINAL OPAQUE {len(receipts)}/{len(sizes['objects'])}", flush=True)
    return "all_downloads_complete_labels_locked"


def run(max_partitions=None):
    if "raincalib-gpu" not in sys.executable:
        raise ValueError("use the specified existing Conda environment")
    os.chdir(ROOT)
    STATE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    config, preflight = read_json(FREEZE), read_json(PREFLIGHT)
    progress_path = STATE / "progress.json"
    with FileLock(STATE / "writer.lock", timeout=0):
        progress = (
            read_json(progress_path)
            if progress_path.exists()
            else {"completed": {}, "attempts": {}, "failures": {}}
        )

        def save(status, **extra):
            progress.update(
                status=status,
                pid=os.getpid(),
                process_created=psutil.Process().create_time(),
                updated_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                test_labels_read=False,
                **extra,
            )
            atomic_json(progress_path, progress)

        try:
            plan_path = STATE / "plan.json"
            tasks = tasks_for(config)
            identity = {
                "freeze_sha256": file_hash(FREEZE),
                "preflight_sha256": file_hash(PREFLIGHT),
                "builder_sha256": file_hash(ROOT / "scripts/build_route_b_pilot.py"),
                "library_sha256": {
                    str(p.relative_to(ROOT)): file_hash(p)
                    for p in sorted((ROOT / "src/cloud2watt").rglob("*.py"))
                },
                "partition_schema": "daily-issue-qc-halo-v1",
            }
            if plan_path.exists():
                if read_json(plan_path)["identity"] != identity:
                    raise ValueError("build identity changed; audit migration before resuming")
            else:
                atomic_json(
                    plan_path,
                    {
                        "identity": identity,
                        "tasks": tasks,
                        "final_raw_download_authorized": True,
                        "final_labels_locked": True,
                        "final_combinations": "2022 final window x all four frozen regions",
                        "qc_halo": "3h before issue day, 6h after, clipped to seasonal support",
                        "formal_training_ready": False,
                    },
                )
            save("planning", total_partitions=len(tasks))
            budget = Budget(config)
            budget.check(256 * 1024**2)
            # Fetch monthly PV just before its first partition; a later-month
            # outage must not block already cached winter development data.
            client = CheckedDevelopmentStore(
                config["satellite_sources"]["2021"]["url"],
                cache_dir=CACHE,
                download_bytes=config["resource_limits"]["cumulative_source_download_bytes"]
                - budget.state["pv_reserved_bytes"]
                - budget.state.get("final_reserved_bytes", 0),
                max_attempts=3,
            )
            if client.metadata_sha256 != config["satellite_sources"]["2021"]["metadata_sha256"]:
                raise ValueError("development source metadata drift")
            inventory_path = STATE / "source-inventory.json"
            if not inventory_path.exists():
                atomic_json(inventory_path, inventory(config, preflight, tasks, client))
            for task_id, info in progress["completed"].items():
                path = DEST / task_id
                if file_hash(path / "verified.json") != info["receipt_sha256"]:
                    raise ValueError("completion receipt changed")
                verify_shard(path)
            done_this_run = 0
            builder = load_builder()
            for task in tasks:
                task_id = task["id"]
                if task_id in progress["completed"]:
                    continue
                output = DEST / task_id
                if (output / "verified.json").exists():
                    verify_shard(output)
                    progress["completed"][task_id] = {
                        "receipt_sha256": file_hash(output / "verified.json")
                    }
                    save("running", current=task_id)
                    continue
                attempt = progress["attempts"].get(task_id, 0)
                if attempt >= 3:
                    continue
                months = pd.period_range(
                    pd.Timestamp(task["start"]).tz_localize(None),
                    (pd.Timestamp(task["end"]) - pd.Timedelta(seconds=1)).tz_localize(None),
                    freq="M",
                )
                names = {
                    f"5_minutely/year={m.year}/month={m.month:02d}/data.parquet" for m in months
                }
                save("preparing_partition", current=task_id)
                try:
                    raw_pv(preflight, budget, names)
                except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
                    progress["failures"][task_id] = {"input_error": type(error).__name__}
                    save("input_partition_deferred", current=task_id)
                    continue
                client.download_limit = (
                    config["resource_limits"]["cumulative_source_download_bytes"]
                    - budget.state["pv_reserved_bytes"]
                )
                cache_now = store_bytes(CACHE)
                budget.check(max(0, 4 * 1024**3 - cache_now) + 128 * 1024**2)
                if output.exists():
                    archived = DEST / "failed" / task_id / f"attempt-{attempt}"
                    if not output.resolve().is_relative_to(
                        DEST.resolve()
                    ) or not archived.resolve().is_relative_to(DEST.resolve()):
                        raise ValueError("unsafe incomplete partition move")
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    if archived.exists():
                        raise FileExistsError(archived)
                    output.rename(archived)
                output.parent.mkdir(parents=True, exist_ok=True)
                progress["attempts"][task_id] = attempt + 1
                save(
                    "running",
                    current=task_id,
                    started_task_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                )
                builder.build(
                    SimpleNamespace(
                        preflight=PREFLIGHT,
                        region=task["region"],
                        window=task["window"],
                        start=task["start"],
                        end=task["end"],
                        issue_start=task["issue_start"],
                        issue_end=task["issue_end"],
                        cache=CACHE,
                        output=output,
                        client=client,
                        local_sources_only=True,
                        data_role="route_b_development_partition",
                        guard=lambda: budget.check(16 * 1024**2),
                    )
                )
                seal_shard(output, task_id)
                progress["completed"][task_id] = {
                    "receipt_sha256": file_hash(output / "verified.json"),
                    "samples": read_json(output / "build_report.json")["samples"],
                }
                progress["failures"].pop(task_id, None)
                save(
                    "running",
                    current=task_id,
                    persistent_bytes=budget.usage(),
                    source_charged_bytes=budget.source_charged(),
                    pv_reserved_bytes=budget.state["pv_reserved_bytes"],
                )
                print(
                    f"PARTITION COMPLETE {task_id} ({len(progress['completed'])}/{len(tasks)})",
                    flush=True,
                )
                done_this_run += 1
                if max_partitions is not None and done_this_run >= max_partitions:
                    save("batch_complete_more_pending")
                    return
            raw_pv(
                preflight,
                budget,
                {n for n in preflight["required_pv_partitions"] if "year=2022/" in n},
            )
            status = (
                final_download(config, budget)
                if len(progress["completed"]) == len(tasks)
                else "partial_failed"
            )
            save(status, current=None)
        except Exception as error:
            progress["failures"][progress.get("current") or "setup"] = {
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            save("failed", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-partitions", type=int)
    run(parser.parse_args().max_partitions)
