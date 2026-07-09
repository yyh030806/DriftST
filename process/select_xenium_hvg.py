"""Select highly variable genes from one Xenium transcript file."""

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transcripts", required=True,
                   help="path to one *_transcripts.parquet file")
    p.add_argument("--candidate_genes", required=True,
                   help="candidate gene list produced by select_xenium_genes.py")
    p.add_argument("--out_file", required=True)
    p.add_argument("--n_top_genes", type=int, default=280)
    p.add_argument("--qv_threshold", type=float, default=20.0)
    p.add_argument("--min_counts", type=int, default=10)
    p.add_argument("--overlaps_nucleus", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    candidates = [line.strip() for line in open(args.candidate_genes) if line.strip()]
    gene2idx = {g: i for i, g in enumerate(candidates)}
    print(f"Candidate genes: {len(candidates)}")

    print("Reading transcripts...")
    cols = ["cell_id", "feature_name", "qv", "overlaps_nucleus"]
    df = pd.read_parquet(args.transcripts, columns=cols)
    df["feature_name"] = df["feature_name"].apply(
        lambda x: x.decode() if isinstance(x, bytes) else x
    )

    if pd.api.types.is_numeric_dtype(df["cell_id"]):
        df = df[df["cell_id"] > 0]
    else:
        df = df[df["cell_id"].astype(str) != "UNASSIGNED"]
    df = df[df["qv"] >= args.qv_threshold]
    if args.overlaps_nucleus:
        df = df[df["overlaps_nucleus"] == 1]
    df = df[df["feature_name"].isin(gene2idx)]
    print(f"Transcripts after filtering: {len(df):,}")

    counts = df.groupby(["cell_id", "feature_name"]).size().reset_index(name="n")
    cell_total = counts.groupby("cell_id")["n"].sum()
    valid_cells = cell_total[cell_total >= args.min_counts].index
    counts = counts[counts["cell_id"].isin(valid_cells)]
    print(f"Valid cells: {len(valid_cells):,} (min_counts={args.min_counts})")

    cell_ids = sorted(valid_cells.tolist())
    cell2idx = {c: i for i, c in enumerate(cell_ids)}
    rows = counts["cell_id"].map(cell2idx).values
    cols_idx = counts["feature_name"].map(gene2idx).values
    vals = counts["n"].values.astype(np.float32)
    X = sp.csr_matrix((vals, (rows, cols_idx)), shape=(len(cell_ids), len(candidates)))

    adata = ad.AnnData(X=X, var=pd.DataFrame(index=candidates))
    print(f"Count matrix: {adata.shape}")

    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_top_genes, flavor="seurat_v3", subset=False
    )
    hvg = adata.var_names[adata.var["highly_variable"]].tolist()
    hvg = sorted(hvg)
    print(f"Selected HVGs: {len(hvg)}")
    print("Top 20:", hvg[:20])

    with open(args.out_file, "w") as f:
        for gene in hvg:
            f.write(gene + "\n")
    print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
