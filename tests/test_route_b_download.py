"""Restart integrity, single-writer ownership and exact daily issue coverage."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock, Timeout

spec = importlib.util.spec_from_file_location(
    "route_b_download", Path(__file__).parents[1] / "scripts/download_route_b.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_daily_shards_cover_each_allowed_issue_once_with_qc_context():
    config = runner.read_json(runner.FREEZE)
    tasks = runner.tasks_for(config)
    assert len(tasks) == 408
    builder = runner.load_builder()
    for window in config["windows"]:
        if window["role"] != "development":
            continue
        pieces = []
        for task in tasks:
            if task["window"] != window["id"] or task["region"] != "r0":
                continue
            _, start, end, issues = builder.pilot_bounds(
                config, "r0", window["id"], task["start"], task["end"]
            )
            selected = issues[
                (issues >= pd.Timestamp(task["issue_start"]))
                & (issues < pd.Timestamp(task["issue_end"]))
            ]
            assert (selected - pd.Timedelta(hours=1) >= start).all()
            assert (selected + pd.Timedelta(hours=4) < end).all()
            pieces.extend(selected.tolist())
        expected = pd.date_range(
            pd.Timestamp(window["start"]) + pd.Timedelta(hours=1),
            pd.Timestamp(window["end"]) - pd.Timedelta(hours=4, minutes=15),
            freq="15min",
        ).tolist()
        assert pieces == expected
    assert not any(t["region"] == "r3" or t["window"] == "final_winter" for t in tasks)


def test_resume_refuses_changed_or_missing_zarr_chunks(tmp_path):
    chunk = tmp_path / "tile"
    chunk.write_bytes(b"verified image bytes")
    digest = runner.file_hash(chunk)
    (tmp_path / "verified.json").write_text(
        json.dumps({"files": {"tile": digest}}), encoding="utf-8"
    )
    runner.verify_shard(tmp_path)
    chunk.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.verify_shard(tmp_path)
    chunk.unlink()
    with pytest.raises(ValueError, match="inventory"):
        runner.verify_shard(tmp_path)


def test_single_writer_lock_refuses_second_owner(tmp_path):
    path = tmp_path / "writer.lock"
    with FileLock(path, timeout=0), pytest.raises(Timeout), FileLock(path, timeout=0):
        pass


def test_budget_stops_before_pv_transfer(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "STATE", tmp_path)
    config = {"resource_limits": {"cumulative_source_download_bytes": 100}}
    budget = runner.Budget(config)
    monkeypatch.setattr(budget, "source_charged", lambda: 80)
    with pytest.raises(ValueError, match="reservation"):
        budget.reserve_pv(10)
    assert not (tmp_path / "transfer.json").exists()


def test_final_download_is_opaque_crc_checked_and_resumable(tmp_path, monkeypatch):
    import base64
    import io
    from types import SimpleNamespace

    import google_crc32c

    state = tmp_path / "state"
    state.mkdir()
    quarantine = tmp_path / "quarantine"
    monkeypatch.setattr(runner, "STATE", state)
    monkeypatch.setattr(runner, "QUARANTINE", quarantine)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("final labels parsed"))
    inventory = state / "source-inventory.json"
    inventory.write_text("{}", encoding="utf-8")
    payload = b"opaque source bytes, never decoded"
    crc = base64.b64encode(google_crc32c.value(payload).to_bytes(4, "big")).decode()
    sizes = {
        "complete": True,
        "inventory_sha256": runner.file_hash(inventory),
        "objects": {
            "data/0.0.0.0": {
                "bytes": len(payload),
                "generation": "123",
                "hash_header": f"crc32c={crc}",
            }
        },
    }
    runner.atomic_json(state / "final-object-sizes.json", sizes)
    limits = {"source_object_bytes": 1000, "cumulative_source_download_bytes": 10000}
    budget = SimpleNamespace(
        state={"pv_reserved_bytes": 0},
        limits=limits,
        source_charged=lambda: 0,
        check=lambda *a: 0,
        path=state / "transfer.json",
    )
    config = {"satellite_sources": {"2022": {"url": "https://example.test/source"}}}
    calls = []

    def request(req, **kwargs):
        calls.append(req.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(runner, "urlopen", request)
    assert runner.final_download(config, budget) == "all_downloads_complete_labels_locked"
    assert "generation=123" in calls[0]
    assert runner.final_download(config, budget) == "all_downloads_complete_labels_locked"
    assert len(calls) == 1
    target = quarantine / "satellite-2022/data/0.0.0.0"
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        runner.final_download(config, budget)


def test_truncated_development_block_is_quarantined_before_decode(tmp_path, monkeypatch):
    store = object.__new__(runner.CheckedDevelopmentStore)
    store.root = tmp_path
    store.objects = tmp_path / "objects"
    key = "data/1.2.3.0"
    path = store._cache_path(key)
    path.parent.mkdir(parents=True)
    good = bytearray(20)
    good[12:16] = (20).to_bytes(4, "little")
    replies = iter([bytes(good[:17]), bytes(good)])

    def fetch(self, key):
        payload = next(replies)
        path.write_bytes(payload)
        return payload

    monkeypatch.setattr(runner.BoundedSeviriStore, "_fetch_bytes", fetch)
    assert store._fetch_bytes(key) == bytes(good)
    assert (tmp_path / "corrupt" / key / "attempt-1").read_bytes() == bytes(good[:17])
    assert runner.read_json(tmp_path / "integrity-failures.json")[key] == 1
    runner.atomic_json(tmp_path / "integrity-failures.json", {key: 3})
    with pytest.raises(RuntimeError, match="integrity retry limit"):
        store._fetch_bytes(key)
