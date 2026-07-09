"""Preprocess HEST-style spot-level datasets for DriftST."""

import json
import logging
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
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
    p.add_argument("--data_dir",   type=str, required=True,
                   help="dataset root containing st/ and wsis/")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gene_list",  type=str, required=True,
                   help="path to selected_gene_list.txt")
    p.add_argument("--neighbor_r", type=float, default=150.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--weights_dir", type=str, default=str(DEFAULT_WEIGHTS_DIR),
                   help="directory containing uni2/ and conch/ weights")
    p.add_argument("--slides",     type=str, nargs="*", default=None,
                   help="slides to process; defaults to SPA119-SPA154")
    p.add_argument("--test_slide", type=str, default=None,
                   help="fixed test slide; leave unset for leave-one-slide-out")
    return p.parse_args()


# ─────────────────────────────────────────────
# Step 1: load gene list
# ─────────────────────────────────────────────

def load_gene_list(gene_list_path: str):
    with open(gene_list_path, "r") as f:
        genes = [l.strip() for l in f if l.strip()]
    logger.info(f"Loaded gene list: {len(genes)} genes ({gene_list_path})")
    return genes


# ─────────────────────────────────────────────
# Step 2: load one slide
# ─────────────────────────────────────────────

def load_one_slide(st_path: Path, slide_id: str, gene_list: list):
    h5ad_file = st_path / f"{slide_id}.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(f"Missing file: {h5ad_file}")

    adata = sc.read_h5ad(str(h5ad_file))

    missing = [g for g in gene_list if g not in adata.var_names]
    if missing:
        logger.warning(f"  [{slide_id}] {len(missing)} genes are missing; filling zeros")

    available = [g for g in gene_list if g in adata.var_names]
    adata_sub = adata[:, available].copy()

    if missing:
        zero_adata = anndata.AnnData(
            X=sp.csr_matrix(
                np.zeros((adata_sub.n_obs, len(missing)), dtype=np.float32)
            ),
            obs=adata_sub.obs.copy(),
            var=pd.DataFrame(index=missing),
        )
        adata_sub = anndata.concat([adata_sub, zero_adata], axis=1, merge="same")
        adata_sub = adata_sub[:, gene_list].copy()
    else:
        adata_sub = adata_sub[:, gene_list].copy()

    if "spatial" not in adata.obsm:
        raise ValueError(f"[{slide_id}] missing obsm['spatial']")
    adata_sub.obsm["spatial"] = adata.obsm["spatial"]

    spot_diameter = None
    spatial_meta  = adata.uns.get("spatial", {})
    if isinstance(spatial_meta, dict):
        for key, val in spatial_meta.items():
            if isinstance(val, dict) and "scalefactors" in val:
                spot_diameter = val["scalefactors"].get("spot_diameter_fullres")
                break
    adata_sub.uns["spot_diameter_fullres"] = float(spot_diameter) if spot_diameter else None

    adata_sub.obs["slide_id"] = slide_id
    adata_sub.obs["pixel_x"]  = adata_sub.obsm["spatial"][:, 0]
    adata_sub.obs["pixel_y"]  = adata_sub.obsm["spatial"][:, 1]
    adata_sub.obs_names       = [f"{slide_id}_{bc}" for bc in adata_sub.obs_names]

    logger.info(f"  [{slide_id}] {adata_sub.n_obs} spots x {adata_sub.n_vars} genes"
                f"  spot_diameter={spot_diameter}")
    return adata_sub


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


# ─────────────────────────────────────────────
# Feature extraction helper: one patch batch -> (B, 2048)
# ─────────────────────────────────────────────

def encode_patches(patches, uni_model, uni_tf, conch_model, conch_tf,
                   uni_ln, conch_ln, device):
    """Extract UNI2+CONCH features for a batch of PIL images."""
    with torch.no_grad():
        uni_t = torch.stack([
            uni_tf(p.resize((224, 224), Image.Resampling.LANCZOS))
            for p in patches
        ]).to(device)
        z_uni = uni_ln(uni_model(uni_t))                           # (B, 1536)

        conch_t = torch.stack([
            conch_tf(p.resize((256, 256), Image.Resampling.LANCZOS))
            for p in patches
        ]).to(device)
        z_conch = conch_ln(
            conch_model.encode_image(conch_t, proj_contrast=False, normalize=False)
        )                                                           # (B, 512)

    return torch.cat([z_uni, z_conch], dim=-1).cpu().numpy()       # (B, 2048)


# ─────────────────────────────────────────────
# Step 4: patch extraction and feature encoding
# ─────────────────────────────────────────────

def extract_features_for_slide(wsi_path: Path, slide_id: str,
                                obs_df: pd.DataFrame,
                                spot_diameter: float,
                                uni_model, uni_tf, conch_model, conch_tf,
                                device: str, batch_size: int) -> dict:
    """Return a mapping from prefixed barcode to a 2048-d image feature."""
    tif_file = wsi_path / f"{slide_id}.tif"
    if not tif_file.exists():
        raise FileNotFoundError(f"Missing WSI: {tif_file}")

    radius = max(112, int(spot_diameter // 2)) if spot_diameter else 112
    logger.info(f"  [{slide_id}] spot_diameter={spot_diameter:.1f}  radius={radius}")

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(str(tif_file))
    w, h  = image.size

    uni_ln   = nn.LayerNorm(1536).to(device)
    conch_ln = nn.LayerNorm(512).to(device)

    barcodes = obs_df.index.tolist()
    results  = {}
    skipped  = 0

    for i in tqdm(range(0, len(barcodes), batch_size),
                  desc=f"  [{slide_id}] feature extraction", leave=False):
        batch_bc = barcodes[i : i + batch_size]
        patches, valid_bc = [], []

        for bc in batch_bc:
            cx = float(obs_df.loc[bc, "pixel_x"])
            cy = float(obs_df.loc[bc, "pixel_y"])
            x0, y0 = cx - radius, cy - radius
            x1, y1 = cx + radius, cy + radius
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                skipped += 1
                continue
            patch = image.crop((x0, y0, x1, y1)).convert("RGB")
            patches.append(patch)
            valid_bc.append(bc)

        if not patches:
            continue

        z = encode_patches(
            patches, uni_model, uni_tf, conch_model, conch_tf,
            uni_ln, conch_ln, device
        )
        for j, bc in enumerate(valid_bc):
            results[bc] = z[j]

    image.close()
    logger.info(f"  [{slide_id}] extracted {len(results)} patches, skipped {skipped} boundary spots")
    return results


# ─────────────────────────────────────────────
# Step 5: spatial neighbors
# ─────────────────────────────────────────────

def build_neighbor_map(adata_all, radius: float) -> dict:
    logger.info(f"Building spatial neighbor index (radius={radius})...")
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
# Step 6: save
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

    # z_img_features.npy: (N_spots, 2048)
    missing = [bc for bc in barcodes if bc not in z_img_all]
    if missing:
        logger.warning(f"  {len(missing)} barcodes are missing image features; filling zeros")

    feat_dim  = next(iter(z_img_all.values())).shape[-1]
    z_img_mat = np.stack([
        z_img_all.get(bc, np.zeros(feat_dim))
        for bc in barcodes
    ]).astype(np.float32)
    np.save(output_dir / "z_img_features.npy", z_img_mat)
    logger.info(f"z_img_features.npy  : {z_img_mat.shape}  (spots × feat_dim)")

    with open(output_dir / "barcodes.json",    "w") as f: json.dump(barcodes,    f)
    with open(output_dir / "gene_names.json",  "w") as f: json.dump(gene_names,  f)
    with open(output_dir / "neighbor_map.json","w") as f: json.dump(neighbor_map, f)
    with open(output_dir / "splits.json",      "w") as f: json.dump(splits, f, indent=2)

    adata_all.obs.to_csv(output_dir / "obs_metadata.csv")

    bc2slide = adata_all.obs["slide_id"].to_dict()
    summary  = {
        "n_spots":     int(gene_matrix.shape[0]),
        "n_genes":     int(gene_matrix.shape[1]),
        "n_slides":    int(len(splits)),
        "feature_dim": int(feat_dim),
        "gene_mean":   float(gene_matrix.mean()),
        "gene_std":    float(gene_matrix.std()),
        "bc2slide":    bc2slide,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("Preprocessing complete. Summary:")
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
    st_path    = data_dir / "st"
    wsi_path   = data_dir / "wsis"
    device     = args.device if torch.cuda.is_available() else "cpu"
    weights_dir = Path(args.weights_dir)

    logger.info(f"Device     : {device}")
    logger.info(f"Data dir   : {data_dir}")
    logger.info(f"Output dir : {output_dir}")

    if args.slides:
        fn_lst = args.slides
    else:
        fn_lst = [f"SPA{i}" for i in range(119, 155)]
    logger.info(f"Slides to process ({len(fn_lst)}): {fn_lst}")

    gene_list = load_gene_list(args.gene_list)

    logger.info("\n=== Step 2: Load slide expression matrices ===")
    adatas_dict: dict = {}
    for slide_id in fn_lst:
        try:
            adatas_dict[slide_id] = load_one_slide(st_path, slide_id, gene_list)
        except Exception as e:
            logger.error(f"  [{slide_id}] failed: {e}")

    if not adatas_dict:
        raise RuntimeError("All slides failed to load")

    adata_all = anndata.concat(
        [adatas_dict[sid] for sid in fn_lst if sid in adatas_dict],
        axis=0, merge="same"
    )
    adata_all.var_names_make_unique()
    logger.info(f"Merged data: {adata_all.shape[0]} spots x {adata_all.shape[1]} genes")

    logger.info("\n=== Step 3: Load encoders ===")
    uni_model, uni_tf, conch_model, conch_tf = load_encoders(device, weights_dir)

    logger.info("\n=== Step 4: Extract image features ===")
    z_img_all = {}
    for slide_id in fn_lst:
        if slide_id not in adatas_dict:
            logger.warning(f"  [{slide_id}] failed in Step 2; skipping feature extraction")
            continue

        mask   = adata_all.obs["slide_id"] == slide_id
        obs_df = adata_all.obs[mask][["pixel_x", "pixel_y"]]
        if len(obs_df) == 0:
            logger.warning(f"  [{slide_id}] has no valid spots; skipping")
            continue

        spot_diam = adatas_dict[slide_id].uns.get("spot_diameter_fullres", None)
        spot_diam = float(spot_diam) if spot_diam else 224.0

        try:
            feats = extract_features_for_slide(
                wsi_path, slide_id, obs_df,
                spot_diam, uni_model, uni_tf,
                conch_model, conch_tf, device, args.batch_size
            )
            z_img_all.update(feats)
        except Exception as e:
            logger.error(f"  [{slide_id}] feature extraction failed: {e}")

    del uni_model, conch_model
    torch.cuda.empty_cache()

    logger.info("\n=== Step 5: Build spatial neighbor index ===")
    neighbor_map = build_neighbor_map(adata_all, radius=args.neighbor_r)

    # ── Step 6: splits ──
    slides = sorted(adata_all.obs["slide_id"].unique().tolist())
    if args.test_slide:
        if args.test_slide not in slides:
            raise ValueError(f"--test_slide {args.test_slide} is not in loaded slides: {slides}")
        logger.info(f"\n=== Step 6: Fixed test slide = {args.test_slide} ===")
        test_mask = adata_all.obs["slide_id"] == args.test_slide
        splits = [{
            "test_slide": args.test_slide,
            "train":      adata_all.obs_names[~test_mask].tolist(),
            "test":       adata_all.obs_names[test_mask].tolist(),
        }]
        logger.info(f"train: {sum(~test_mask)} spots, test: {sum(test_mask)} spots")
    else:
        logger.info("\n=== Step 6: Generate leave-one-slide-out splits ===")
        splits = []
        for test_slide in slides:
            test_mask = adata_all.obs["slide_id"] == test_slide
            splits.append({
                "test_slide": test_slide,
                "train":      adata_all.obs_names[~test_mask].tolist(),
                "test":       adata_all.obs_names[test_mask].tolist(),
            })
        logger.info(f"Generated {len(splits)} folds")

    logger.info("\n=== Step 7: Save processed data ===")
    save_all(output_dir, adata_all, z_img_all, neighbor_map, splits, gene_list)


if __name__ == "__main__":
    main()
