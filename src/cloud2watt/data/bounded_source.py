"""Dedicated, bounded HTTP object cache for route B source builds."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cloud2watt.data.seviri import SeviriStore
from cloud2watt.run_state import atomic_json


class BoundedSeviriStore(SeviriStore):
    """Single-process writer; serializes fetches and evicts only owned objects.

    The byte budget is cumulative across restarts. A request reserves its full
    maximum size before network access, including failed attempts, so crashes
    cannot reset the transfer budget. Other processes must use another cache.
    """

    def __init__(
        self,
        store_url,
        *,
        cache_dir,
        cache_bytes=4 * 1024**3,
        download_bytes=100 * 1024**3,
        object_bytes=16 * 1024**2,
        reserve_fraction=0.2,
        **kwargs,
    ):
        if not 0 <= reserve_fraction < 1 or not 0 < object_bytes <= cache_bytes:
            raise ValueError("invalid cache limits")
        if download_bytes < object_bytes:
            raise ValueError("transfer budget smaller than one object")
        self.root = Path(cache_dir).resolve()
        self.marker = self.root / "owner.json"
        self.lock = threading.RLock()
        self.cache_limit, self.download_limit = cache_bytes, download_bytes
        self.object_limit, self.reserve_fraction = object_bytes, reserve_fraction
        self.evicted_bytes = 0
        self.root.mkdir(parents=True, exist_ok=True)
        if self.marker.exists():
            self.state = json.loads(self.marker.read_text(encoding="utf-8"))
            if self.state.get("schema") != "c2w-bounded-source-v1" or self.state.get(
                "url"
            ) != store_url.rstrip("/"):
                raise ValueError("cache ownership/source mismatch")
        else:
            if any(self.root.iterdir()):
                raise ValueError("refusing to adopt a nonempty unowned cache")
            self.state = {
                "schema": "c2w-bounded-source-v1",
                "url": store_url.rstrip("/"),
                "charged_bytes": 0,
            }
            atomic_json(self.marker, self.state)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        # Partial files from interrupted writes are owned objects too.
        self._trim(0)
        super().__init__(store_url, cache_dir=self.objects, **kwargs)

    def _cache_path(self, key):
        if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", key):
            raise ValueError("invalid source object key")
        if any(part in {".", ".."} for part in key.split("/")):
            raise ValueError("invalid source object key")
        path = self.objects.joinpath(*key.split("/"))
        if not path.resolve().is_relative_to(self.objects.resolve()):
            raise ValueError("source object escapes cache")
        return path

    def _trim(self, incoming):
        files = list(self.objects.rglob("*")) if self.objects.exists() else []
        if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in files):
            raise ValueError("cache must not contain links")
        files = sorted((p for p in files if p.is_file()), key=lambda p: p.stat().st_mtime_ns)
        size = sum(p.stat().st_size for p in files)
        for path in files:
            if size + incoming <= self.cache_limit:
                break
            if not path.resolve().is_relative_to(self.objects.resolve()):
                raise ValueError("unsafe cache eviction path")
            amount = path.stat().st_size
            path.unlink()
            size -= amount
            self.evicted_bytes += amount
        if size + incoming > self.cache_limit:
            raise ValueError("object exceeds cache capacity")
        return size

    def _fetch_bytes(self, key):
        with self.lock:
            path = self._cache_path(key)
            if path.exists():
                self.cache_hits += 1
                path.touch()
                return path.read_bytes()
            request = Request(
                f"{self.store_url}/{key}", headers={"User-Agent": "cloud2watt-route-b/2.0"}
            )
            for attempt in range(1, self.max_attempts + 1):
                if self.state["charged_bytes"] + self.object_limit > self.download_limit:
                    raise ValueError("cumulative source transfer budget exhausted")
                self._trim(self.object_limit)
                disk = shutil.disk_usage(self.root)
                if disk.free - self.object_limit < disk.total * self.reserve_fraction:
                    raise ValueError("source cache would cross disk reserve")
                self.state["charged_bytes"] += self.object_limit
                atomic_json(self.marker, self.state)
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        payload = response.read(self.object_limit + 1)
                    if len(payload) > self.object_limit:
                        raise ValueError("source object exceeds bounded read limit")
                    break
                except (HTTPError, URLError, TimeoutError):
                    if attempt == self.max_attempts:
                        raise
                    self.retry_count += 1
                    time.sleep(2 ** (attempt - 1))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
            # Successful requests are charged their actual bytes; failed/crashed
            # requests conservatively retain the full reservation.
            self.state["charged_bytes"] -= self.object_limit - len(payload)
            atomic_json(self.marker, self.state)
            self.downloaded_bytes += len(payload)
            self.network_fetches += 1
            return payload
