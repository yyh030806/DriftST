"""Preprocess Xenium cell-level data for DriftST.

This script reconstructs a cell-by-gene count matrix from transcript parquet
files, extracts UNI2+CONCH H&E patch features, builds a spatial neighbor map,
and writes the processed_data format used by DriftST.
"""

import json
import logging
import argparse
import os
import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from sklearn.neighbors import KDTree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_DIR = Path(os.environ.get(
    "DRIFTST_WEIGHTS_DIR",
    Path(__file__).resolve().parents[1] / "weights",
))


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gene_list",  type=str, required=True,
                   help="path to selected_gene_list.txt")
    p.add_argument("--slides",     type=str, nargs="+", default=["TENX94", "TENX95"])
    p.add_argument("--test_slide", type=str, default="TENX95")
    p.add_argument("--neighbor_r", type=float, default=300.0,
                   help="neighbor search radius in H&E WSI pixels")
    p.add_argument("--context_r",  type=int,   default=112,
                   help="patch radius in pixels; 112 yields a 224x224 patch")
    p.add_argument("--min_counts", type=int,   default=10,
                   help="minimum transcripts per cell")
    p.add_argument("--qv_threshold", type=float, default=20.0,
                   help="transcript quality threshold")
    p.add_argument("--overlaps_nucleus", action="store_true",
                   help="keep only transcripts with overlaps_nucleus=1")
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--device",     type=str,   default="cuda")
    p.add_argument("--weights_dir", type=str, default=str(DEFAULT_WEIGHTS_DIR),
                   help="directory containing uni2/ and conch/ weights")
    p.add_argument("--max_cells",  type=int,   default=None,
                   help="optional random downsampling cap per slide")
    p.add_argument("--seed",       type=int,   default=42,
                   help="random seed")
    return p.parse_args()


# ─────────────────────────────────────────────
# Step 1: gene list
# ─────────────────────────────────────────────

def load_gene_list(path: str) -> list:
    with open(path) as f:
        genes = [l.strip() for l in f if l.strip()]
    logger.info(f"Target gene list: {len(genes)} genes")
    return genes


# ─────────────────────────────────────────────
# Step 2: reconstruct single-cell expression from transcript parquet files
# ─────────────────────────────────────────────

def load_one_slide(data_dir: Path, slide_id: str, gene_list: list,
                   qv_threshold: float, min_counts: int,
                   overlaps_nucleus: bool = False,
                   max_cells: int = None, seed: int = 42) -> anndata.AnnData:
    """Reconstruct a cell-by-gene count matrix from transcript records."""
    transcript_path = data_dir / "transcripts" / f"{slide_id}_transcripts.parquet"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript file: {transcript_path}")

    logger.info(f"  [{slide_id}] reading transcript parquet...")
    cols = ["cell_id", "feature_name", "he_x", "he_y", "qv", "overlaps_nucleus"]
    df = pd.read_parquet(str(transcript_path), columns=cols)

    # Decode byte-encoded gene names.
    df["feature_name"] = df["feature_name"].apply(
        lambda x: x.decode() if isinstance(x, bytes) else x
    )

    # Remove unassigned cells and low-quality transcripts.
    if pd.api.types.is_numeric_dtype(df["cell_id"]):
        df = df[df["cell_id"] > 0]
    else:
        df = df[df["cell_id"].astype(str) != "UNASSIGNED"]
    df = df[df["qv"] >= qv_threshold]

    if overlaps_nucleus:
        before = len(df)
        df = df[df["overlaps_nucleus"] == 1]
        logger.info(f"  [{slide_id}] overlaps_nucleus filter: {before:,} -> {len(df):,} transcripts")

    logger.info(f"  [{slide_id}] transcripts after filtering: {len(df):,}")

    logger.info(f"  [{slide_id}] building expression matrix...")
    counts = df.groupby(["cell_id", "feature_name"]).size().reset_index(name="n")

    cell_total = counts.groupby("cell_id")["n"].sum()
    valid_cells = cell_total[cell_total >= min_counts].index
    counts = counts[counts["cell_id"].isin(valid_cells)]
    logger.info(f"  [{slide_id}] valid cells: {len(valid_cells):,} (min_counts={min_counts})")

    if max_cells is not None and len(valid_cells) > max_cells:
        rng = np.random.default_rng(seed)
        valid_cells = pd.Index(rng.choice(valid_cells, size=max_cells, replace=False))
        counts = counts[counts["cell_id"].isin(set(valid_cells))]
        logger.info(f"  [{slide_id}] downsampled to {max_cells} cells (seed={seed})")

    cell_ids  = sorted(valid_cells.tolist())
    cell2idx  = {c: i for i, c in enumerate(cell_ids)}
    gene2idx  = {g: i for i, g in enumerate(gene_list)}

    counts_target = counts[counts["feature_name"].isin(gene2idx)]
    rows = counts_target["cell_id"].map(cell2idx).values
    cols = counts_target["feature_name"].map(gene2idx).values
    vals = counts_target["n"].values.astype(np.float32)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(cell_ids), len(gene_list)))

    logger.info(f"  [{slide_id}] computing cell centroids in H&E coordinates...")
    cell_df_all = df[df["cell_id"].isin(set(cell_ids))]
    centroid = cell_df_all.groupby("cell_id")[["he_x", "he_y"]].mean()
    centroid = centroid.reindex(cell_ids)

    obs = pd.DataFrame({
        "slide_id": slide_id,
        "pixel_x":  centroid["he_x"].values,
        "pixel_y":  centroid["he_y"].values,
    }, index=[f"{slide_id}_{c}" for c in cell_ids])

    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=gene_list),
    )
    logger.info(f"  [{slide_id}] {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


# ─────────────────────────────────────────────
# Step 3: load encoders
# ─────────────────────────────────────────────

def load_encoders(device: str, weights_dir: Path):
    from timm.models.vision_transformer import VisionTransformer
    from timm.layers.mlp import GluMlp
    from torchvision import transforms

    uni2_weights_dir = weights_dir / "uni2"
    conch_weights_dir = weights_dir / "conch"

    logger.info(f"Loading UNI2 from {uni2_weights_dir}")
    ckpt_files = (list(uni2_weights_dir.glob("*.bin")) +
                  list(uni2_weights_dir.glob("*.safetensors")) +
                  list(uni2_weights_dir.glob("*.pth")))
    if not ckpt_files:
        raise FileNotFoundError(f"No UNI2 checkpoint found in {uni2_weights_dir}")
    ckpt_path  = ckpt_files[0]

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
    else:
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]

    uni_model = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1536, depth=24, num_heads=24,
        mlp_ratio=16/3, qkv_bias=True, init_values=1e-5, num_classes=0,
        no_embed_class=True, reg_tokens=8, mlp_layer=GluMlp, act_layer=nn.SiLU,
    )
    uni_model.load_state_dict(state_dict, strict=True)
    uni_model.eval().to(device)

    uni_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    from conch.open_clip_custom import create_model_from_pretrained
    logger.info(f"Loading CONCH from {conch_weights_dir}")
    conch_model, conch_tf = create_model_from_pretrained(
        "conch_ViT-B-16", str(conch_weights_dir / "pytorch_model.bin")
    )
    conch_model.eval().to(device)

    return uni_model, uni_tf, conch_model, conch_tf


def encode_patches(patches, uni_model, uni_tf, conch_model, conch_tf,
                   uni_ln, conch_ln, device):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        uni_t = torch.stack([
            uni_tf(p.resize((224, 224), Image.Resampling.BILINEAR))
            for p in patches
        ]).to(device)
        z_uni = uni_ln(uni_model(uni_t).float())

        conch_t = torch.stack([
            conch_tf(p.resize((256, 256), Image.Resampling.BILINEAR))
            for p in patches
        ]).to(device)
        z_conch = conch_ln(
            conch_model.encode_image(conch_t, proj_contrast=False, normalize=False).float()
        )

    return torch.cat([z_uni, z_conch], dim=-1).cpu().numpy()  # (B, 2048)


# ─────────────────────────────────────────────
# Step 4: patch extraction
# ─────────────────────────────────────────────

def extract_features_for_slide(data_dir: Path, slide_id: str,
                                obs_df: pd.DataFrame,
                                context_r: int,
                                uni_model, uni_tf, conch_model, conch_tf,
                                device: str, batch_size: int) -> dict:
    """
    Return a mapping from barcode to a 2048-d UNI2+CONCH feature.
    """
    import tifffile
    wsi_path = data_dir / "wsis" / f"{slide_id}.tif"
    if not wsi_path.exists():
        raise FileNotFoundError(f"Missing WSI: {wsi_path}")

    logger.info(f"  [{slide_id}] loading WSI into memory: {wsi_path.name}, radius={context_r}px")
    with tifffile.TiffFile(str(wsi_path)) as tif:
        arr = tif.pages[0].asarray()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] != 3:
        arr = np.concatenate([arr] * 3, axis=-1)
    h, w = arr.shape[:2]
    logger.info(f"  [{slide_id}] WSI shape: {arr.shape}, dtype={arr.dtype}")

    uni_ln   = nn.LayerNorm(1536).to(device)
    conch_ln = nn.LayerNorm(512).to(device)

    barcodes = obs_df.index.tolist()
    results  = {}
    skipped  = 0

    for i in tqdm(range(0, len(barcodes), batch_size),
                  desc=f"  [{slide_id}] patch extraction", leave=False):
        batch_bc = barcodes[i : i + batch_size]
        patches, valid_bc = [], []

        for bc in batch_bc:
            cx = float(obs_df.loc[bc, "pixel_x"])
            cy = float(obs_df.loc[bc, "pixel_y"])
            x0 = int(cx - context_r)
            y0 = int(cy - context_r)
            x1 = int(cx + context_r)
            y1 = int(cy + context_r)
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                skipped += 1
                continue
            patch = Image.fromarray(arr[y0:y1, x0:x1]).convert("RGB")
            patches.append(patch)
            valid_bc.append(bc)

        if not patches:
            continue

        z = encode_patches(patches, uni_model, uni_tf, conch_model, conch_tf,
                           uni_ln, conch_ln, device)
        for j, bc in enumerate(valid_bc):
            results[bc] = z[j]

    del arr
    logger.info(f"  [{slide_id}] extracted {len(results)} patches, skipped {skipped} boundary cells")
    return results


# ─────────────────────────────────────────────
# Step 5: neighbor graph
# ─────────────────────────────────────────────

def build_neighbor_map(adata_all: anndata.AnnData, radius: float) -> dict:
    logger.info(f"Building spatial neighbor index (radius={radius} px)...")
    neighbor_map = {}

    for slide_id in adata_all.obs["slide_id"].unique():
        mask     = adata_all.obs["slide_id"] == slide_id
        sub      = adata_all[mask]
        barcodes = sub.obs_names.tolist()
        coords   = sub.obs[["pixel_x", "pixel_y"]].values.astype(float)

        tree = KDTree(coords)
        for i, bc in enumerate(barcodes):
            idxs = tree.query_radius([coords[i]], r=radius)[0]
            idxs = idxs[idxs != i]
            neighbor_map[bc] = [barcodes[j] for j in idxs]

    n_avg = np.mean([len(v) for v in neighbor_map.values()])
    logger.info(f"  Average neighbors: {n_avg:.2f}")
    return neighbor_map


# ─────────────────────────────────────────────
# Step 6: save processed data
# ─────────────────────────────────────────────

def save_all(output_dir: Path, adata_all, z_img_all: dict,
             neighbor_map: dict, splits: list, gene_names: list):
    output_dir.mkdir(parents=True, exist_ok=True)

    barcodes    = adata_all.obs_names.tolist()
    gene_matrix = adata_all.X
    if sp.issparse(gene_matrix):
        gene_matrix = gene_matrix.toarray()
    gene_matrix = gene_matrix.astype(np.float32)

    np.save(output_dir / "gene_expression.npy", gene_matrix)
    logger.info(f"gene_expression.npy : {gene_matrix.shape}")

    missing  = [bc for bc in barcodes if bc not in z_img_all]
    if missing:
        logger.warning(f"  {len(missing)} barcodes are missing image features; filling zeros")

    feat_dim  = next(iter(z_img_all.values())).shape[-1] if z_img_all else 2048
    z_img_mat = np.stack([
        z_img_all.get(bc, np.zeros(feat_dim))
        for bc in barcodes
    ]).astype(np.float32)
    np.save(output_dir / "z_img_features.npy", z_img_mat)
    logger.info(f"z_img_features.npy  : {z_img_mat.shape}")

    with open(output_dir / "barcodes.json",    "w") as f: json.dump(barcodes,    f)
    with open(output_dir / "gene_names.json",  "w") as f: json.dump(gene_names,  f)
    with open(output_dir / "neighbor_map.json","w") as f: json.dump(neighbor_map, f)
    with open(output_dir / "splits.json",      "w") as f: json.dump(splits, f, indent=2)

    adata_all.obs.to_csv(output_dir / "obs_metadata.csv")

    bc2slide = adata_all.obs["slide_id"].to_dict()
    summary  = {
        "n_cells":     int(gene_matrix.shape[0]),
        "n_genes":     int(gene_matrix.shape[1]),
        "n_slides":    int(len(splits)),
        "feature_dim": int(feat_dim),
        "gene_mean":   float(gene_matrix.mean()),
        "gene_std":    float(gene_matrix.std()),
        "data_type":   "xenium_single_cell",
        "bc2slide":    bc2slide,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    for k, v in summary.items():
        if k != "bc2slide":
            logger.info(f"  {k}: {v}")
    logger.info(f"Output directory: {output_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device     = args.device if torch.cuda.is_available() else "cpu"

    logger.info(f"Device      : {device}")
    logger.info(f"Data dir    : {data_dir}")
    logger.info(f"Slides      : {args.slides}")
    logger.info(f"Neighbor r  : {args.neighbor_r} px")
    logger.info(f"Context r   : {args.context_r} px -> patch {2*args.context_r}x{2*args.context_r}")
    weights_dir = Path(args.weights_dir)

    gene_list = load_gene_list(args.gene_list)

    logger.info("\n=== Step 2: Reconstruct single-cell expression matrix ===")
    adatas_dict = {}
    for slide_id in args.slides:
        try:
            adatas_dict[slide_id] = load_one_slide(
                data_dir, slide_id, gene_list,
                qv_threshold=args.qv_threshold,
                min_counts=args.min_counts,
                overlaps_nucleus=args.overlaps_nucleus,
                max_cells=args.max_cells,
                seed=args.seed,
            )
        except Exception as e:
            logger.error(f"  [{slide_id}] failed: {e}")

    if not adatas_dict:
        raise RuntimeError("All slides failed to load")

    adata_all = anndata.concat(
        [adatas_dict[sid] for sid in args.slides if sid in adatas_dict],
        axis=0, merge="same"
    )
    adata_all.var_names_make_unique()
    logger.info(f"Merged data: {adata_all.shape[0]:,} cells x {adata_all.shape[1]} genes")

    logger.info("\n=== Step 3: Load image encoders ===")
    uni_model, uni_tf, conch_model, conch_tf = load_encoders(device, weights_dir)

    logger.info("\n=== Step 4: Extract image patch features ===")
    z_img_all = {}
    for slide_id in args.slides:
        if slide_id not in adatas_dict:
            continue
        mask   = adata_all.obs["slide_id"] == slide_id
        obs_df = adata_all.obs[mask][["pixel_x", "pixel_y"]]
        try:
            feats = extract_features_for_slide(
                data_dir, slide_id, obs_df,
                context_r=args.context_r,
                uni_model=uni_model, uni_tf=uni_tf,
                conch_model=conch_model, conch_tf=conch_tf,
                device=device, batch_size=args.batch_size,
            )
            z_img_all.update(feats)
        except Exception as e:
            logger.error(f"  [{slide_id}] feature extraction failed: {e}", exc_info=True)

    del uni_model, conch_model
    torch.cuda.empty_cache()

    logger.info("\n=== Step 5: Build spatial neighbor index ===")
    neighbor_map = build_neighbor_map(adata_all, radius=args.neighbor_r)

    # ── Step 6: splits ──
    slides = sorted(adata_all.obs["slide_id"].unique().tolist())
    if args.test_slide:
        if args.test_slide not in slides:
            raise ValueError(f"--test_slide {args.test_slide} is not in {slides}")
        test_mask = adata_all.obs["slide_id"] == args.test_slide
        splits = [{
            "test_slide": args.test_slide,
            "train":      adata_all.obs_names[~test_mask].tolist(),
            "test":       adata_all.obs_names[test_mask].tolist(),
        }]
        logger.info(f"test={args.test_slide}: train={sum(~test_mask):,}, test={sum(test_mask):,}")
    else:
        splits = []
        for ts in slides:
            tm = adata_all.obs["slide_id"] == ts
            splits.append({
                "test_slide": ts,
                "train":      adata_all.obs_names[~tm].tolist(),
                "test":       adata_all.obs_names[tm].tolist(),
            })

    logger.info("\n=== Step 7: Save processed data ===")
    save_all(output_dir, adata_all, z_img_all, neighbor_map, splits, gene_list)


if __name__ == "__main__":
    main()
