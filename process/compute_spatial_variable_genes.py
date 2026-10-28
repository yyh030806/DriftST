"""Rank spatially variable genes with Moran's I."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import barcode_fingerprint, processed_data_fingerprint, validate_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--slide", type=str, default="TENX94")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--splits_file", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--fallback_train_slide", action="store_true",
                   help="use another training slide if the requested slide has too few training cells")
    p.add_argument("--n_neighbors", type=int, default=6,
                   help="number of neighbors for the spatial graph")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("Loading processed data...")
    X = np.load(data_dir / "gene_expression.npy", mmap_mode="r")
    with open(data_dir / "barcodes.json") as f:
        barcodes = json.load(f)
    with open(data_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    obs = pd.read_csv(data_dir / "obs_metadata.csv", index_col=0, low_memory=False)
    obs = obs.loc[barcodes]

    splits_path = Path(args.splits_file) if args.splits_file else data_dir / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"fold={args.fold} out of range")
    split = splits[args.fold]
    validate_split(split, barcodes)
    train_bcs = set(split["train"])
    train_mask = obs.index.isin(train_bcs)
    slide = args.slide
    idxs = np.where((obs["slide_id"].values == slide) & train_mask)[0]
    if len(idxs) < 3 and args.fallback_train_slide:
        counts = obs.loc[train_mask, "slide_id"].value_counts()
        eligible = sorted((s for s, count in counts.items() if count >= 3), key=str)
        if eligible:
            slide = eligible[0]
            if isinstance(slide, np.generic):
                slide = slide.item()
            idxs = np.where((obs["slide_id"].values == slide) & train_mask)[0]
            print(f"Requested SVG slide {args.slide} has too few training cells; using {slide}")
    if len(idxs) < 3:
        raise ValueError(f"slide '{slide}' needs at least 3 training cells in fold {args.fold}")
    print(f"Slide {slide}, fold {args.fold}: {len(idxs)} training cells")

    X_slide = X[idxs]
    obs_slide = obs.iloc[idxs].copy()
    adata = ad.AnnData(
        X=np.log1p(X_slide),
        obs=obs_slide,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = obs_slide[["pixel_x", "pixel_y"]].values

    n_neighbors = min(args.n_neighbors, len(idxs) - 1)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    print(f"Building k={n_neighbors} spatial graph...")
    sq.gr.spatial_neighbors(
        adata, coord_type="generic", n_neighs=n_neighbors, spatial_key="spatial"
    )

    print("Computing Moran's I...")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names.tolist())
    moran = adata.uns["moranI"].sort_values("I", ascending=False)

    ranking = moran.index.tolist()
    scores = moran["I"].to_dict()
    print("Computing processed-data fingerprint (reading input files)...")
    out = {
        "slide": slide,
        "fold": args.fold,
        "method": "moranI",
        "training_split_sha256": barcode_fingerprint(split["train"]),
        "data_sha256": processed_data_fingerprint(data_dir),
        "ranking": ranking,
        "scores": {g: float(scores[g]) for g in ranking},
    }

    out_path = Path(args.output) if args.output else data_dir / f"svg_ranking_fold{args.fold}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nTop 20 SVGs (Moran's I):")
    for i, g in enumerate(ranking[:20]):
        print(f"  {i + 1:2d}. {g:<20s} I={scores[g]:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
