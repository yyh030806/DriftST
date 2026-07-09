"""Generate spatial 5-fold cross-validation splits for one Xenium slide."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--slide", type=str, default="TENX94",
                   help="slide ID used for spatial cross-validation")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--axis", type=str, choices=["x", "y"], default="y",
                   help="spatial split axis; y produces horizontal bands")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    obs = pd.read_csv(data_dir / "obs_metadata.csv", index_col=0)
    with open(data_dir / "barcodes.json") as f:
        barcodes = json.load(f)
    obs = obs.loc[barcodes]

    slide_obs = obs[obs["slide_id"] == args.slide].copy()
    if len(slide_obs) == 0:
        raise ValueError(
            f"slide '{args.slide}' not found; available slides: "
            f"{obs['slide_id'].unique().tolist()}"
        )

    coord_col = "pixel_y" if args.axis == "y" else "pixel_x"
    slide_obs = slide_obs.sort_values(coord_col)
    slide_bcs = slide_obs.index.to_numpy()
    n = len(slide_bcs)

    print(f"Slide: {args.slide} | cells: {n} | split axis: {coord_col}")
    print(f"{coord_col} range: {slide_obs[coord_col].min():.1f} - {slide_obs[coord_col].max():.1f}")

    bands = np.array_split(np.arange(n), args.n_folds)
    splits = []
    all_bcs = np.array(barcodes)
    other_bcs = all_bcs[obs["slide_id"].values != args.slide].tolist()

    for fold_id, test_idx in enumerate(bands):
        test_bcs = slide_bcs[test_idx].tolist()
        train_bcs = other_bcs + [bc for bc in slide_bcs.tolist() if bc not in set(test_bcs)]
        lo = float(slide_obs.iloc[test_idx][coord_col].min())
        hi = float(slide_obs.iloc[test_idx][coord_col].max())
        splits.append({
            "test_slide": args.slide,
            "fold": fold_id,
            "train": train_bcs,
            "test": test_bcs,
            "spatial_info": {
                "axis": coord_col,
                "coord_lo": lo,
                "coord_hi": hi,
            },
        })
        print(f"  Fold {fold_id}: train={len(train_bcs)}, test={len(test_bcs)}, band={lo:.1f}-{hi:.1f}")

    out_path = data_dir / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWrote {out_path} ({args.n_folds} folds)")


if __name__ == "__main__":
    main()
