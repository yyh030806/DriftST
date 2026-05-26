"""
select_xenium_hvg.py
====================
从 Xenium transcripts 选 top-N 高变基因(HVG)。

流程:
  1. 用与 preprocess_xenium.py 完全一致的过滤重建 cell×gene 计数矩阵
     (cell_id>0, qv>=qv_threshold, overlaps_nucleus==1, 细胞 min_counts)
  2. 候选基因 = select_xenium_genes.py 输出的真实基因(已排除控制探针)
  3. scanpy seurat_v3(直接吃 raw counts)选 n_top_genes 个 HVG
  4. 按字母排序写出 selected_gene_list.txt(覆盖)

用法:
  python select_xenium_hvg.py \
      --transcripts .../xenium_coad/transcripts/TENX111_transcripts.parquet \
      --candidate_genes .../xenium_coad/processed_data/selected_gene_list.txt \
      --out_file .../xenium_coad/processed_data/selected_gene_list.txt \
      --n_top 280
"""
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
import scanpy as sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transcripts", required=True, help="单个 *_transcripts.parquet 路径")
    p.add_argument("--candidate_genes", required=True,
                   help="候选真实基因列表(select_xenium_genes.py 输出)")
    p.add_argument("--out_file", required=True)
    p.add_argument("--n_top", type=int, default=280)
    p.add_argument("--qv_threshold", type=float, default=20.0)
    p.add_argument("--min_counts", type=int, default=70)
    p.add_argument("--overlaps_nucleus", action="store_true", default=True)
    args = p.parse_args()

    with open(args.candidate_genes) as f:
        candidates = [l.strip() for l in f if l.strip()]
    print(f"候选真实基因: {len(candidates)}")

    print("读取 transcripts ...")
    df = pd.read_parquet(args.transcripts,
                         columns=["cell_id", "feature_name", "qv", "overlaps_nucleus"])
    df["feature_name"] = df["feature_name"].apply(
        lambda x: x.decode() if isinstance(x, bytes) else x)

    # 与 preprocess_xenium.py 完全一致的过滤(cell_id 可能是数值或字符串)
    if pd.api.types.is_numeric_dtype(df["cell_id"]):
        df = df[df["cell_id"] > 0]
    else:
        df = df[df["cell_id"].astype(str) != "UNASSIGNED"]
    df = df[df["qv"] >= args.qv_threshold]
    if args.overlaps_nucleus:
        df = df[df["overlaps_nucleus"] == 1]
    df = df[df["feature_name"].isin(set(candidates))]
    print(f"过滤后转录本: {len(df):,}")

    counts = df.groupby(["cell_id", "feature_name"]).size().reset_index(name="n")
    cell_total = counts.groupby("cell_id")["n"].sum()
    valid_cells = cell_total[cell_total >= args.min_counts].index
    counts = counts[counts["cell_id"].isin(valid_cells)]
    print(f"有效细胞: {len(valid_cells):,} (min_counts={args.min_counts})")

    cell_ids = sorted(valid_cells.tolist())
    cell2idx = {c: i for i, c in enumerate(cell_ids)}
    gene2idx = {g: i for i, g in enumerate(candidates)}
    rows = counts["cell_id"].map(cell2idx).values
    cols = counts["feature_name"].map(gene2idx).values
    vals = counts["n"].values.astype(np.float32)
    X = sp.csr_matrix((vals, (rows, cols)),
                      shape=(len(cell_ids), len(candidates)))

    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=candidates))
    print(f"计数矩阵: {adata.shape}")

    # seurat_v3 直接吃 raw counts
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top, flavor="seurat_v3")
    hvg = adata.var_names[adata.var["highly_variable"]].tolist()
    hvg = sorted(hvg)
    print(f"选出 HVG: {len(hvg)} 个")
    print("前 20:", hvg[:20])

    with open(args.out_file, "w") as f:
        f.write("\n".join(hvg) + "\n")
    print(f"已保存: {args.out_file}")


if __name__ == "__main__":
    main()
