"""
generate_xenium_5fold_splits.py
================================
基于现有预处理数据，生成 GHIST 风格的 5-fold 空间交叉验证 splits.json。

策略：
  - 只使用单张 slide（默认 TENX94）的细胞
  - 按 pixel_y（H&E 图像 y 坐标）升序排列，等分为 5 个水平带（spatial fold）
  - 每折：1 个带作为 test，其余 4 个带作为 train
  - 不需要重新跑图像特征提取，直接复用现有 processed_data

输出：
  processed_data/splits.json（覆盖原文件）

用法：
  python generate_xenium_5fold_splits.py \
      --data_dir /data/buyonggan/DriftST/hest1k_datasets/xenium_janesick/processed_data \
      --slide TENX94 \
      --n_folds 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="/data/buyonggan/DriftST/hest1k_datasets/xenium_janesick/processed_data")
    p.add_argument("--slide",   type=str, default="TENX94",
                   help="用于 5-fold CV 的单张 slide ID")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--axis",    type=str, default="y", choices=["x", "y"],
                   help="空间切分轴，GHIST 默认按 y 轴水平切分")
    return p.parse_args()


def main():
    args    = parse_args()
    data_dir = Path(args.data_dir)

    obs = pd.read_csv(data_dir / "obs_metadata.csv", index_col=0)
    slide_obs = obs[obs["slide_id"] == args.slide].copy()

    if len(slide_obs) == 0:
        raise ValueError(f"slide '{args.slide}' 不在 obs_metadata.csv 中，"
                         f"可用 slide: {obs['slide_id'].unique().tolist()}")

    coord_col = "pixel_y" if args.axis == "y" else "pixel_x"
    slide_obs = slide_obs.sort_values(coord_col)
    barcodes  = slide_obs.index.tolist()
    n         = len(barcodes)

    print(f"Slide: {args.slide}  |  cells: {n}  |  split axis: {coord_col}")
    print(f"{coord_col} range: {slide_obs[coord_col].min():.1f} – {slide_obs[coord_col].max():.1f}")

    # 等分 n_folds 个空间带
    fold_indices = np.array_split(np.arange(n), args.n_folds)

    splits = []
    for fold_id, test_idx in enumerate(fold_indices):
        test_bcs  = [barcodes[i] for i in test_idx]
        train_bcs = [barcodes[i] for i in range(n) if i not in set(test_idx)]

        coord_lo = slide_obs.iloc[test_idx[0]][coord_col]
        coord_hi = slide_obs.iloc[test_idx[-1]][coord_col]

        splits.append({
            "fold":       fold_id,
            "test_slide": args.slide,
            "train":      train_bcs,
            "test":       test_bcs,
            "spatial_info": {
                "axis":     coord_col,
                "coord_lo": float(coord_lo),
                "coord_hi": float(coord_hi),
            },
        })
        print(f"  Fold {fold_id}: train={len(train_bcs)}, test={len(test_bcs)}, "
              f"{coord_col}=[{coord_lo:.0f}, {coord_hi:.0f}]")

    out_path = data_dir / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nsplits.json 写入 → {out_path}  ({args.n_folds} folds)")


if __name__ == "__main__":
    main()
