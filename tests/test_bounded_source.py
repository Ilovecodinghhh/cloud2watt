"""Resource and ownership failures must precede further source downloads."""

import io
import json

import pytest

from cloud2watt.data.bounded_source import BoundedSeviriStore

META = json.dumps({"metadata": {"data/.zarray": {}}}).encode()


def client(tmp_path, monkeypatch, **kwargs):
    calls = []

    def fetch(request, **unused):
        calls.append(request.full_url)
        return io.BytesIO(META if request.full_url.endswith(".zmetadata") else b"x" * 90)

    monkeypatch.setattr("cloud2watt.data.bounded_source.urlopen", fetch)
    store = BoundedSeviriStore(
        "https://example.test/source",
        cache_dir=tmp_path / "cache",
        cache_bytes=200,
        object_bytes=100,
        download_bytes=1000,
        reserve_fraction=0,
        **kwargs,
    )
    return store, calls


def test_eviction_and_restart_budget(tmp_path, monkeypatch):
    store, calls = client(tmp_path, monkeypatch)
    store._fetch_bytes("data/0")
    store._fetch_bytes("data/1")
    store._fetch_bytes("data/1")
    assert len(calls) == 3
    assert store.cache_hits == 1
    assert sum(p.stat().st_size for p in store.objects.rglob("*") if p.is_file()) <= 200
    assert store.evicted_bytes > 0
    charged = store.state["charged_bytes"]
    again, _ = client(tmp_path, monkeypatch)
    assert again.state["charged_bytes"] >= charged
    again.download_limit = again.state["charged_bytes"]
    with pytest.raises(ValueError, match="budget"):
        again._fetch_bytes("data/2")
    assert not (again.objects / "data/2").exists()


def test_unowned_and_traversal_refused(tmp_path, monkeypatch):
    store, calls = client(tmp_path, monkeypatch)
    before = len(calls)
    for key in ("../outside", "data/../../outside", "C:/outside", "data\\outside", "/outside"):
        with pytest.raises(ValueError):
            store._fetch_bytes(key)
    assert len(calls) == before
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "keep").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unowned"):
        BoundedSeviriStore("https://example.test", cache_dir=unowned)
    assert (unowned / "keep").read_text(encoding="utf-8") == "keep"


def test_oversized_response_never_persisted(tmp_path, monkeypatch):
    store, _ = client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "cloud2watt.data.bounded_source.urlopen", lambda *a, **k: io.BytesIO(b"x" * 101)
    )
    with pytest.raises(ValueError, match="bounded read"):
        store._fetch_bytes("data/oversized")
    assert not (store.objects / "data/oversized").exists()
