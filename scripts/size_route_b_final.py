"""HEAD-only final satellite storage inventory; never reads labels or image payloads."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from filelock import FileLock

from cloud2watt.run_state import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "outputs/project-v2/route-b-download"


def main():
    config = json.loads(
        (ROOT / "configs/project-v2/route-b-freeze.json").read_text(encoding="utf-8")
    )
    inventory_path = STATE / "source-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    output = STATE / "final-object-sizes.json"
    with FileLock(STATE / "size-final.lock", timeout=0):
        data = (
            json.loads(output.read_text(encoding="utf-8"))
            if output.exists()
            else {
                "inventory_sha256": file_hash(inventory_path),
                "objects": {},
                "failures": {},
                "attempts": {},
                "method": "HEAD_only",
                "final_labels_read": False,
            }
        )
        if data["inventory_sha256"] != file_hash(inventory_path):
            raise ValueError("source inventory changed")
        source = config["satellite_sources"]["2022"]["url"]

        def size(key):
            with urlopen(Request(f"{source}/{key}", method="HEAD"), timeout=30) as response:
                amount = int(response.headers["Content-Length"])
                if amount <= 0 or amount > config["resource_limits"]["source_object_bytes"]:
                    raise ValueError("invalid final object size")
                return {
                    "bytes": amount,
                    "etag": response.headers.get("ETag"),
                    "generation": response.headers.get("x-goog-generation"),
                    "hash_header": response.headers.get("x-goog-hash"),
                }

        keys = [
            key
            for key in inventory["final_satellite_keys"]
            if key not in data["objects"] and data["attempts"].get(key, 0) < 3
        ]
        # Charge attempts durably before issuing requests, including interrupted HEADs.
        # Batches bound the crash penalty and the number of queued network operations.
        for first in range(0, len(keys), 64):
            batch = keys[first : first + 64]
            for key in batch:
                data["attempts"][key] = data["attempts"].get(key, 0) + 1
            atomic_json(output, data)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(size, key): key for key in batch}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        data["objects"][key] = future.result()
                        data["failures"].pop(key, None)
                    except Exception as error:
                        data["failures"][key] = type(error).__name__
            data["verified_object_count"] = len(data["objects"])
            data["total_object_count"] = len(inventory["final_satellite_keys"])
            data["verified_bytes"] = sum(v["bytes"] for v in data["objects"].values())
            data["complete"] = data["verified_object_count"] == data["total_object_count"]
            atomic_json(output, data)
            print(
                f"HEAD {data['verified_object_count']}/{data['total_object_count']}: "
                f"{data['verified_bytes']} bytes",
                flush=True,
            )


if __name__ == "__main__":
    main()
