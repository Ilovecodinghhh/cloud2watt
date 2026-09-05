"""Plot a real IR_108 frame pair and its estimated global cloud motion."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from cloud2watt.optical_flow import phase_correlation_translation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/paired-30d-v1"))
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("outputs/optical-flow-example.png"))
    args = parser.parse_args()
    samples = pd.read_parquet(args.data / "sample_index.parquet")
    row = samples.iloc[args.sample]
    first_index, second_index = [int(value) for value in row["satellite_frame_indices"][-2:]]
    frames = zarr.open_group(args.data / "satellite_frames.zarr", mode="r")["frames"]
    first = np.asarray(frames[int(row["site_index"]), first_index, 2], dtype=np.float32)
    second = np.asarray(frames[int(row["site_index"]), second_index, 2], dtype=np.float32)
    dy, dx = phase_correlation_translation(first, second, downsample=4)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, image, title in zip(axes, (first, second), ("IR_108 t-15", "IR_108 issue time"),
                                  strict=True):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    center = first.shape[0] / 2
    axes[1].arrow(center, center, dx * 4, dy * 4, color="red", width=1.0,
                  head_width=5, length_includes_head=True)
    figure.suptitle(f"site={row['site_id']}  estimated motion: dy={dy:.1f}, dx={dx:.1f} px")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
