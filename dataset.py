"""
dataset.py
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SpatialDataset(Dataset):
    """
    Single-split (train or test) spatial transcriptomics dataset.

    Args:
        data_dir      : directory produced by preprocess.py
        barcodes      : spot barcodes belonging to this split
        max_neighbors : maximum 1-hop neighbors to return (Visium: up to 6)
    """

    def __init__(
        self,
        data_dir: str,
        barcodes: list[str],
        max_neighbors: int = 6,
    ):
        self.data_dir      = Path(data_dir)
        self.max_neighbors = max_neighbors

        # ── 全量数组：mmap 加载，多 worker 不额外占内存 ──────────────────────
        gene_mat  = np.load(self.data_dir / "gene_expression.npy", mmap_mode='r')  # (N, G)
        z_img_mat = np.load(self.data_dir / "z_img_features.npy",  mmap_mode='r')  # (N, D)

        with open(self.data_dir / "barcodes.json") as f:
            all_barcodes: list[str] = json.load(f)
        with open(self.data_dir / "neighbor_map.json") as f:
            neighbor_map: dict[str, list[str]] = json.load(f)

        bc2idx = {bc: i for i, bc in enumerate(all_barcodes)}

        # ── 过滤当前 split 的 barcode，保留顺序 ───────────────────────────────
        valid_bcs = [bc for bc in barcodes if bc in bc2idx]
        idxs      = [bc2idx[bc] for bc in valid_bcs]

        self.barcodes  = valid_bcs
        self.n_spots   = len(valid_bcs)
        self.n_genes   = gene_mat.shape[1]
        self.feat_dim  = z_img_mat.shape[1]

        # 当前 split 的特征（常驻内存，只有 split 大小）
        self.gene_expr = torch.tensor(gene_mat[idxs],  dtype=torch.float32)  # (N_split, G)
        self.z_img     = torch.tensor(z_img_mat[idxs], dtype=torch.float32)  # (N_split, D)

        # 全量数组保留引用，供邻居索引用（mmap，不复制）
        self._all_gene = gene_mat   # (N_all, G)
        self._all_zimg = z_img_mat  # (N_all, D)
        self._bc2idx   = bc2idx

        # ── slide_id：从 barcode 前缀解析（格式假设为 "slideXX_ACGTACGT..."）
        # 如果 barcode 没有 slide 前缀，改为从 summary.json 的 bc2slide 字段读
        self.slide_ids = self._parse_slide_ids(valid_bcs, self.data_dir)

        # ── 预计算邻居索引矩阵（在 __init__ 里跑一次，__getitem__ 直接用）────
        # neighbor_idxs[i, k] = 全量数组里的绝对索引；-1 表示 padding
        K = max_neighbors
        self.neighbor_idxs = torch.full(
            (self.n_spots, K), fill_value=-1, dtype=torch.long
        )
        for i, bc in enumerate(valid_bcs):
            nbs = neighbor_map.get(bc, [])
            for k, nbc in enumerate(nbs[:K]):
                if nbc in bc2idx:
                    self.neighbor_idxs[i, k] = bc2idx[nbc]

    # ── 辅助：解析 slide_id ───────────────────────────────────────────────────

    @staticmethod
    def _parse_slide_ids(barcodes: list[str], data_dir: Path) -> list[str]:
        """
        优先从 summary.json 的 bc2slide 字段读；
        否则从 barcode 前缀（slideXX_...）解析；
        再否则统一返回 'unknown'。
        """
        summary_path = data_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            bc2slide = summary.get("bc2slide", {})
            if bc2slide:
                return [bc2slide.get(bc, "unknown") for bc in barcodes]

        # 尝试从 barcode 前缀解析（格式：slideXX_ACGT）
        slide_ids = []
        for bc in barcodes:
            parts = bc.split("_", 1)
            slide_ids.append(parts[0] if len(parts) == 2 else "unknown")
        return slide_ids

    # ── Dataset 接口 ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_spots

    def __getitem__(self, idx: int) -> dict:
        gene_expr = self.gene_expr[idx]   # (G,)
        z_img     = self.z_img[idx]       # (D,)

        K = self.max_neighbors
        G = self.n_genes
        D = self.feat_dim

        # 邻居索引（预计算好的）
        nidxs = self.neighbor_idxs[idx]   # (K,)，-1 = padding
        valid  = nidxs >= 0               # (K,) bool mask

        neighbor_genes = torch.zeros(K, G, dtype=torch.float32)
        neighbor_zimg  = torch.zeros(K, D, dtype=torch.float32)

        if valid.any():
            valid_nidxs = nidxs[valid].numpy()
            neighbor_genes[valid] = torch.tensor(
                self._all_gene[valid_nidxs], dtype=torch.float32
            )
            neighbor_zimg[valid] = torch.tensor(
                self._all_zimg[valid_nidxs], dtype=torch.float32
            )

        return {
            "z_img":          z_img,            # (D,)
            "gene_expr":      gene_expr,         # (G,)
            "neighbor_genes": neighbor_genes,    # (K, G)，padding = 0
            "neighbor_zimg":  neighbor_zimg,     # (K, D)，padding = 0
            "neighbor_valid": valid,             # (K,) bool，True = 有效邻居
            "slide_id":       self.slide_ids[idx],
            "barcode":        self.barcodes[idx],
        }


# ── 构建 train/test split ─────────────────────────────────────────────────────

def build_datasets(
    data_dir:      str,
    fold:          int = 0,
    max_neighbors: int = 6,
) -> tuple[SpatialDataset, SpatialDataset, dict]:
    """
    Load a leave-one-slide-out fold from splits.json.

    Returns:
        train_ds, test_ds, meta
    """
    data_dir = Path(data_dir)

    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    with open(data_dir / "summary.json") as f:
        summary = json.load(f)

    if fold >= len(splits):
        raise ValueError(f"fold={fold} out of range; only {len(splits)} slides")

    split    = splits[fold]
    train_ds = SpatialDataset(data_dir, split["train"], max_neighbors)
    test_ds  = SpatialDataset(data_dir, split["test"],  max_neighbors)

    meta = {
        "n_genes":     summary["n_genes"],
        "feature_dim": summary["feature_dim"],
        "test_slide":  split["test_slide"],
        "n_train":     len(train_ds),
        "n_test":      len(test_ds),
        "n_folds":     len(splits),
    }
    return train_ds, test_ds, meta