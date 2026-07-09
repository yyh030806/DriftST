"""Dataset utilities for preprocessed DriftST data.

``z_img_features.npy`` may be stored as either ``(N, D)`` or
``(N, n_versions, D)``. The 3D format stores the original feature at index 0
and augmented versions at later indices; training samples one version at random
while evaluation always uses the original feature.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SpatialDataset(Dataset):
    """
    Single-split (train or test) spatial transcriptomics dataset.

    Args:
        data_dir      : processed_data directory produced by preprocessing
        barcodes      : cell or spot barcodes belonging to this split
        max_neighbors : maximum 1-hop neighbors to return (Visium: up to 6)
        is_train      : True samples augmented features; False uses originals.
    """

    def __init__(
        self,
        data_dir: str,
        barcodes: list[str],
        max_neighbors: int = 6,
        is_train: bool = True,
    ):
        self.data_dir      = Path(data_dir)
        self.max_neighbors = max_neighbors
        self.is_train      = is_train

        gene_mat  = np.load(self.data_dir / "gene_expression.npy", mmap_mode='r')  # (N, G)
        z_img_raw = np.load(self.data_dir / "z_img_features.npy",  mmap_mode='r')

        if z_img_raw.ndim == 3:
            self.n_versions = z_img_raw.shape[1]
            self.feat_dim   = z_img_raw.shape[2]
            self.has_aug    = True
        else:
            self.n_versions = 1
            self.feat_dim   = z_img_raw.shape[1]
            self.has_aug    = False

        with open(self.data_dir / "barcodes.json") as f:
            all_barcodes: list[str] = json.load(f)
        with open(self.data_dir / "neighbor_map.json") as f:
            neighbor_map: dict[str, list[str]] = json.load(f)

        bc2idx = {bc: i for i, bc in enumerate(all_barcodes)}

        valid_bcs = [bc for bc in barcodes if bc in bc2idx]
        idxs      = [bc2idx[bc] for bc in valid_bcs]

        self.barcodes  = valid_bcs
        self.n_spots   = len(valid_bcs)
        self.n_genes   = gene_mat.shape[1]

        self.gene_counts = torch.tensor(gene_mat[idxs], dtype=torch.float32)
        self.gene_expr = torch.log1p(self.gene_counts)

        if self.has_aug:
            self.z_img = torch.tensor(
                z_img_raw[idxs], dtype=torch.float32
            )
        else:
            self.z_img = torch.tensor(
                z_img_raw[idxs], dtype=torch.float32
            )

        # Keep the full mmap array for neighbor feature lookup.
        self._all_zimg = z_img_raw

        # slide_id
        self.slide_ids = self._parse_slide_ids(valid_bcs, self.data_dir)

        K = max_neighbors
        self.neighbor_idxs = torch.full(
            (self.n_spots, K), fill_value=-1, dtype=torch.long
        )
        for i, bc in enumerate(valid_bcs):
            nbs = neighbor_map.get(bc, [])
            for k, nbc in enumerate(nbs[:K]):
                if nbc in bc2idx:
                    self.neighbor_idxs[i, k] = bc2idx[nbc]

    @staticmethod
    def _parse_slide_ids(barcodes: list[str], data_dir: Path) -> list[str]:
        summary_path = data_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            bc2slide = summary.get("bc2slide", {})
            if bc2slide:
                return [bc2slide.get(bc, "unknown") for bc in barcodes]

        slide_ids = []
        for bc in barcodes:
            parts = bc.split("_", 1)
            slide_ids.append(parts[0] if len(parts) == 2 else "unknown")
        return slide_ids

    def __len__(self) -> int:
        return self.n_spots

    def __getitem__(self, idx: int) -> dict:
        gene_expr = self.gene_expr[idx]   # (G,)

        if self.has_aug:
            if self.is_train:
                aug_idx = random.randint(0, self.n_versions - 1)
            else:
                aug_idx = 0
            z_img = self.z_img[idx, aug_idx]
        else:
            z_img = self.z_img[idx]

        K = self.max_neighbors
        D = self.feat_dim

        nidxs = self.neighbor_idxs[idx]
        valid = nidxs >= 0

        neighbor_zimg = torch.zeros(K, D, dtype=torch.float32)

        if valid.any():
            valid_nidxs = nidxs[valid].numpy()
            if self.has_aug:
                nb_aug_idx = aug_idx if self.is_train else 0
                neighbor_zimg[valid] = torch.tensor(
                    self._all_zimg[valid_nidxs, nb_aug_idx], dtype=torch.float32
                )
            else:
                neighbor_zimg[valid] = torch.tensor(
                    self._all_zimg[valid_nidxs], dtype=torch.float32
                )

        return {
            "z_img":            z_img,
            "gene_expr":        gene_expr,
            "gene_counts":      self.gene_counts[idx],
            "neighbor_zimg":    neighbor_zimg,
            "neighbor_valid":   valid,
            "slide_id":         self.slide_ids[idx],
            "barcode":          self.barcodes[idx],
        }


def build_datasets(
    data_dir:      str,
    fold:          int = 0,
    max_neighbors: int = 6,
) -> tuple[SpatialDataset, SpatialDataset, dict]:
    data_dir = Path(data_dir)

    with open(data_dir / "splits.json") as f:
        splits = json.load(f)
    with open(data_dir / "summary.json") as f:
        summary = json.load(f)

    if fold >= len(splits):
        raise ValueError(f"fold={fold} out of range; only {len(splits)} slides")

    split    = splits[fold]
    train_ds = SpatialDataset(data_dir, split["train"], max_neighbors, is_train=True)
    test_ds  = SpatialDataset(data_dir, split["test"],  max_neighbors, is_train=False)

    meta = {
        "n_genes":     summary["n_genes"],
        "feature_dim": summary["feature_dim"],
        "test_slide":  split["test_slide"],
        "n_train":     len(train_ds),
        "n_test":      len(test_ds),
        "n_folds":     len(splits),
    }
    return train_ds, test_ds, meta
