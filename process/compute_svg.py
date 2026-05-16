"""
compute_svg.py
==============
用 Moran's I 计算 Xenium 数据中的 SVG 排序，保存到 processed_data/svg_ranking.json。

只使用 TENX94 的细胞（5-fold CV 的目标 slide）。
输出：
  svg_ranking.json  —  按 Moran's I 降序排列的基因名列表（全部 200 个基因）

用法：
  python compute_svg.py \
      --data_dir /data/buyonggan/DriftST/hest1k_datasets/xenium_janesick/processed_data \
      --slide TENX94
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import anndata
import squidpy as sq
from pathlib import Path

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="/data/buyonggan/DriftST/hest1k_datasets/xenium_janesick/processed_data")
    p.add_argument("--slide", type=str, default="TENX94")
    p.add_argument("--n_neighbors", type=int, default=10,
                   help="构建空间邻居图时的 k 近邻数")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("加载预处理数据 ...")
    gene_expr = np.load(data_dir / "gene_expression.npy", mmap_mode="r")
    obs = pd.read_csv(data_dir / "obs_metadata.csv", index_col=0)

    with open(data_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    with open(data_dir / "barcodes.json") as f:
        all_barcodes = json.load(f)

    # 只取目标 slide
    mask = obs["slide_id"] == args.slide
    slide_obs = obs[mask]
    bc2idx = {bc: i for i, bc in enumerate(all_barcodes)}
    idxs = [bc2idx[bc] for bc in slide_obs.index if bc in bc2idx]

    print(f"Slide {args.slide}: {len(idxs)} 个细胞")

    X = gene_expr[idxs].copy()  # (N, G)
    coords = slide_obs[["pixel_x", "pixel_y"]].values[
        [i for i, bc in enumerate(slide_obs.index) if bc in bc2idx]
    ]

    # 构建 AnnData
    adata = anndata.AnnData(
        X=np.log1p(X).astype(np.float32),
        obs=pd.DataFrame(index=slide_obs.index[
            [i for i, bc in enumerate(slide_obs.index) if bc in bc2idx]
        ]),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = coords.astype(np.float32)

    print(f"构建 k={args.n_neighbors} 空间邻居图 ...")
    sq.gr.spatial_neighbors(adata, n_neighs=args.n_neighbors, coord_type="generic",
                            spatial_key="spatial")

    print("计算 Moran's I ...")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=gene_names, n_jobs=8)

    moranI = adata.uns["moranI"]
    moranI_sorted = moranI.sort_values("I", ascending=False)

    ranking = moranI_sorted.index.tolist()
    scores  = moranI_sorted["I"].tolist()

    out = {
        "slide":   args.slide,
        "method":  "moranI",
        "ranking": ranking,          # 全部基因，按 Moran's I 降序
        "scores":  {g: float(s) for g, s in zip(ranking, scores)},
    }

    out_path = data_dir / "svg_ranking.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nTop 20 SVGs (Moran's I):")
    for i, (g, s) in enumerate(zip(ranking[:20], scores[:20])):
        print(f"  {i+1:2d}. {g:<20s}  I={s:.4f}")
    print(f"\nsvg_ranking.json → {out_path}")


if __name__ == "__main__":
    main()
