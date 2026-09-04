"""Probe gated access to Open Climate Fix's UK PV dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud2watt.data.probes import (
    ProbeError,
    ProbeReport,
    RemoteFile,
    choose_smallest_file,
    redact_secrets,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="openclimatefix/uk_pv")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--partition-prefix", default="5_minutely/year=2020/")
    parser.add_argument("--download-partition", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/probes/uk_pv.json"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    columns = reader.fieldnames or []
    required = {"ss_id", "latitude_rounded", "longitude_rounded", "kWp"}
    missing = sorted(required.difference(columns))
    if missing:
        raise ProbeError(f"metadata.csv is missing expected columns: {missing}")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "columns": columns,
    }


def run(args: argparse.Namespace) -> ProbeReport:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ProbeError(
            "HF_TOKEN is not set. Accept the uk_pv access conditions in a browser, "
            "create a read token, and export HF_TOKEN without committing it."
        )

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ProbeError('Install data dependencies with: pip install -e ".[data]"') from exc

    api = HfApi(token=token)
    info = api.dataset_info(
        repo_id=args.repo_id,
        revision=args.revision,
        files_metadata=True,
        token=token,
    )
    files = [RemoteFile(path=item.rfilename, size=item.size) for item in info.siblings]
    sample = choose_smallest_file(files, prefix=args.partition_prefix, suffix=".parquet")

    def download_small_file(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=info.sha,
                token=token,
            )
        )

    metadata_path = download_small_file("metadata.csv")
    bad_data_path = download_small_file("bad_data.csv")
    details: dict[str, Any] = {
        "repo_id": args.repo_id,
        "revision": info.sha,
        "gated": info.gated,
        "file_count": len(files),
        "metadata": summarize_metadata(metadata_path),
        "bad_data": {
            "path": str(bad_data_path),
            "size_bytes": bad_data_path.stat().st_size,
            "sha256": sha256_file(bad_data_path),
        },
        "smallest_partition": {"path": sample.path, "size_bytes": sample.size},
        "partition_downloaded": False,
    }

    if args.download_partition:
        partition_path = download_small_file(sample.path)
        details["partition_downloaded"] = True
        details["partition"] = {
            "path": str(partition_path),
            "size_bytes": partition_path.stat().st_size,
            "sha256": sha256_file(partition_path),
        }

        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ProbeError('Install PyArrow with: pip install -e ".[data]"') from exc
        parquet_file = parquet.ParquetFile(partition_path)
        details["partition"].update(
            {
                "rows": parquet_file.metadata.num_rows,
                "row_groups": parquet_file.metadata.num_row_groups,
                "columns": parquet_file.schema.names,
            }
        )

    return ProbeReport.create("uk_pv", "ok", details)


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:  # CLI boundary: persist a sanitized diagnostic for every failure.
        message = redact_secrets(str(exc))
        report = ProbeReport.create("uk_pv", "blocked", {"error": message})
        write_report(args.output, report)
        print(f"UK PV probe blocked: {message}", file=sys.stderr)
        return 2

    write_report(args.output, report)
    print(f"UK PV probe succeeded; report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
