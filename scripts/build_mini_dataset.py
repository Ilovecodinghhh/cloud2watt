"""Build a deterministic 20-site, seven-day UK PV mini dataset."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cloud2watt.data.mini import build_power_mini_dataset, write_power_mini_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=Path, required=True, help="5-minute UK PV Parquet")
    parser.add_argument("--metadata", type=Path, required=True, help="UK PV metadata.csv")
    parser.add_argument("--bad-data", type=Path, help="Optional UK PV bad_data.csv")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument(
        "--power-value-semantics",
        choices=("interval_energy", "instantaneous_power"),
        required=True,
    )
    parser.add_argument("--start", default="2020-12-01", help="UTC start date (inclusive)")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sites", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/processed/mini-v1"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(
                f"output already exists: {args.output}; pass --overwrite to replace it"
            )
        allowed_root = (Path.cwd() / "data" / "processed").resolve()
        resolved_output = args.output.resolve()
        if allowed_root not in resolved_output.parents:
            raise SystemExit("--overwrite is restricted to a child of data/processed")
        shutil.rmtree(args.output)
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    parquet_start = pd.Timestamp(start)
    power = pd.read_parquet(
        args.power,
        filters=[
            ("datetime_GMT", ">", parquet_start),
            ("datetime_GMT", "<=", parquet_start + pd.Timedelta(days=args.days)),
        ],
    )
    metadata = pd.read_csv(args.metadata)
    bad_periods = pd.read_csv(args.bad_data) if args.bad_data else None
    derived, sites, manifest = build_power_mini_dataset(
        power,
        metadata,
        start_utc=start,
        days=args.days,
        site_count=args.sites,
        bad_periods=bad_periods,
        source_revision=args.source_revision,
        power_value_semantics=args.power_value_semantics,
    )
    write_power_mini_dataset(args.output, derived, sites, manifest)
    print(f"mini dataset written to {args.output} ({len(derived)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
