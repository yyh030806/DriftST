"""Rank spatially variable genes with Moran's I."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--slide", type=str, default="TENX94")
    p.add_argument("--n_neighbors", type=int, default=6,
                   help="number of neighbors for the spatial graph")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("Loading processed data...")
    X = np.load(data_dir / "gene_expression.npy")
    with open(data_dir / "barcodes.json") as f:
        barcodes = json.load(f)
    with open(data_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    obs = pd.read_csv(data_dir / "obs_metadata.csv", index_col=0)
    obs = obs.loc[barcodes]

    idxs = np.where(obs["slide_id"].values == args.slide)[0]
    if len(idxs) == 0:
        raise ValueError(f"slide '{args.slide}' not found in obs_metadata.csv")
    print(f"Slide {args.slide}: {len(idxs)} cells")

    X_slide = X[idxs]
    obs_slide = obs.iloc[idxs].copy()
    adata = ad.AnnData(
        X=np.log1p(X_slide),
        obs=obs_slide,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = obs_slide[["pixel_x", "pixel_y"]].values

    print(f"Building k={args.n_neighbors} spatial graph...")
    sq.gr.spatial_neighbors(
        adata, coord_type="generic", n_neighs=args.n_neighbors, spatial_key="spatial"
    )

    print("Computing Moran's I...")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names.tolist())
    moran = adata.uns["moranI"].sort_values("I", ascending=False)

    ranking = moran.index.tolist()
    scores = moran["I"].to_dict()
    out = {
        "slide": args.slide,
        "method": "moranI",
        "ranking": ranking,
        "scores": {g: float(scores[g]) for g in ranking},
    }

    out_path = data_dir / "svg_ranking.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nTop 20 SVGs (Moran's I):")
    for i, g in enumerate(ranking[:20]):
        print(f"  {i + 1:2d}. {g:<20s} I={scores[g]:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
