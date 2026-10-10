"""Atomic experiment artifacts and complete epoch-boundary recovery state."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from cloud2watt.evaluation.forecast import PROTOCOL

CHECKPOINT_VERSION = 2


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def atomic_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
                             + "\n", encoding="utf-8")


def atomic_torch(path, value):
    with atomic_path(path) as temporary:
        torch.save(value, temporary)


@contextmanager
def run_lock(path):
    """OS lock releases on process death; a leftover file is not a stale lock."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".run.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"run is active: {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def environment_snapshot():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                if d.metadata.get("Name")}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted(packages.items())), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def source_snapshot(root):
    paths = [*Path(root).glob("src/**/*.py"), *Path(root).glob("scripts/*.py"),
             Path(root) / "pyproject.toml"]
    # Content includes uncommitted changes; Git HEAD alone is not an execution identity.
    return {p.relative_to(root).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(paths)}


def capture_rng(loaders):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loaders": {k: v.generator.get_state() for k, v in loaders.items()}}


def restore_rng(state, loaders):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    for key, loader in loaders.items():
        loader.generator.set_state(state["loaders"][key].cpu())


def load_epoch(path, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("legacy/incomplete checkpoint cannot resume under V2")
    if payload.get("identity") != identity:
        raise ValueError("resume identity mismatch (config/data/split/statistics/code/environment)")
    required = {"model_state", "best_model_state", "optimizer_state", "scaler_state",
                "rng", "history", "epoch", "best_epoch", "best_score", "stale_epochs",
                "sampler_epoch"}
    if not required <= payload.keys():
        raise ValueError("checkpoint lacks complete epoch recovery state")
    if [x["epoch"] for x in payload["history"]] != list(range(1, payload["epoch"] + 1)):
        raise ValueError("checkpoint has missing epoch history")
    return payload


REQUIRED_ARTIFACTS = ("best.pt", "latest.pt", "metrics.json", "predictions.parquet",
                      "provenance.json", "split_manifest.json", "training_history.csv",
                      "config.json", "config.yaml", "environment.json", "statistics.json",
                      "source.json", "source.zip")


def mark_complete(path, identity):
    path = Path(path)
    hashes = {name: file_hash(path / name) for name in REQUIRED_ARTIFACTS}
    atomic_json(path / "complete.json", {"protocol": PROTOCOL, "identity": identity,
                                         "artifacts": hashes})


def is_complete(path, identity):
    path = Path(path)
    try:
        record = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity or record.get("protocol") != PROTOCOL:
            return False
        if set(record["artifacts"]) != set(REQUIRED_ARTIFACTS):
            return False
        return all(file_hash(path / name) == value for name, value in record["artifacts"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False
