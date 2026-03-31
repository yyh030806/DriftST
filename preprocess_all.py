"""
preprocess_all.py
=================
多 slide 合并预处理脚本，适用于 HER2ST 数据集。
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from PIL import Image
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

WEIGHTS_DIR       = Path("/data/buyonggan/DriftST/weights")
UNI2_WEIGHTS_DIR  = WEIGHTS_DIR / "uni2"
CONCH_WEIGHTS_DIR = WEIGHTS_DIR / "conch"


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True,
                   help="visium_format/ 目录，下面有 A1/ A2/ ... 子目录")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录，所有 slide 合并后保存在这里")
    p.add_argument("--n_hvg",      type=int,   default=200,
                   help="HVG 数量（默认 200）")
    p.add_argument("--patch_size", type=int,   default=224)
    p.add_argument("--neighbor_r", type=float, default=150.0)
    p.add_argument("--min_genes",  type=int,   default=200)
    p.add_argument("--min_cells",  type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=64,
                   help="图像特征提取的 batch size")
    p.add_argument("--device",     type=str,   default="cuda")
    p.add_argument("--slides",     type=str,   nargs="*", default=None,
                   help="指定处理哪些 slide，默认处理全部")
    return p.parse_args()


# ─────────────────────────────────────────────
# Step 1: 单个 slide 的基因表达预处理
# ─────────────────────────────────────────────

def load_one_slide_expr(slide_dir: Path, slide_id: str,
                        min_genes: int, min_cells: int):
    """
    加载单个 slide 的基因表达，只做 QC 和归一化，不做 HVG（后面全局做）。

    Returns: AnnData（obs 里有 slide_id / pixel_x / pixel_y 列）
    """
    h5_path = slide_dir / "filtered_feature_bc_matrix.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"找不到 h5 文件: {h5_path}")

    adata = sc.read_10x_h5(str(h5_path))

    # 加载空间坐标
    spatial_dir = slide_dir / "spatial"
    pos_path    = spatial_dir / "tissue_positions_list.csv"
    if not pos_path.exists():
        pos_path = spatial_dir / "tissue_positions.csv"

    pos_df = pd.read_csv(pos_path)
    if "barcode" not in pos_df.columns:
        pos_df.columns = ["barcode", "in_tissue", "array_row", "array_col",
                          "pixel_y", "pixel_x"]
    if "pxl_col_in_fullres" in pos_df.columns:
        pos_df = pos_df.rename(columns={
            "pxl_col_in_fullres": "pixel_x",
            "pxl_row_in_fullres": "pixel_y",
        })
    pos_df = pos_df.set_index("barcode")
    pos_df = pos_df[pos_df["in_tissue"] == 1]

    common = adata.obs_names.intersection(pos_df.index)
    adata  = adata[common].copy()
    adata.obsm["spatial"] = pos_df.loc[
        adata.obs_names, ["pixel_x", "pixel_y"]
    ].values.astype(float)

    # QC
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 归一化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 标记 slide 信息，barcode 加 slide 前缀防止跨 slide 碰撞
    adata.obs["slide_id"] = slide_id
    adata.obs["pixel_x"]  = pos_df.loc[adata.obs_names, "pixel_x"].values
    adata.obs["pixel_y"]  = pos_df.loc[adata.obs_names, "pixel_y"].values
    adata.obs_names       = [f"{slide_id}_{bc}" for bc in adata.obs_names]

    logger.info(f"  [{slide_id}] {adata.shape[0]} spots x {adata.shape[1]} genes (after QC)")
    return adata


# ─────────────────────────────────────────────
# Step 2: 全局 HVG 对齐
# ─────────────────────────────────────────────

def global_hvg(adatas: list, n_hvg: int):
    """
    把多个 slide 的 AnnData 拼起来，做全局 HVG 筛选。
    所有 slide 统一用同一套 HVG，保证基因维度对齐。

    Returns: (adata_all, gene_names)
    """
    import anndata
    logger.info("合并所有 slide 做全局 HVG 筛选...")

    # 取公共基因
    common_genes = adatas[0].var_names
    for a in adatas[1:]:
        common_genes = common_genes.intersection(a.var_names)
    logger.info(f"  公共基因数: {len(common_genes)}")

    adatas_sub = [a[:, common_genes].copy() for a in adatas]
    adata_all  = anndata.concat(adatas_sub, axis=0, merge="same")
    adata_all.var_names_make_unique()

    sc.pp.highly_variable_genes(adata_all, n_top_genes=n_hvg)
    hvg_names = adata_all.var_names[adata_all.var.highly_variable].tolist()
    adata_all  = adata_all[:, hvg_names].copy()

    logger.info(f"  HVG 筛选后: {adata_all.shape[0]} spots x {adata_all.shape[1]} genes")
    return adata_all, hvg_names


# ─────────────────────────────────────────────
# Step 3: HE patch 提取
# ─────────────────────────────────────────────

def load_he_image(img_path: Path) -> np.ndarray:
    suffix = img_path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg"]:
        return np.array(Image.open(img_path).convert("RGB"))
    import tifffile as tiff
    img = tiff.imread(str(img_path))
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img


def extract_patches_for_slide(slide_dir: Path, slide_id: str,
                               obs_df: pd.DataFrame, patch_size: int):
    """
    obs_df: index = prefixed barcode（slide_id_barcode），
            columns: pixel_x, pixel_y
    Returns: {prefixed_barcode: np.ndarray (H, W, 3)}
    """
    he_path = slide_dir / "spatial" / "tissue_hires_image.png"
    if not he_path.exists():
        he_path = slide_dir / "spatial" / "tissue_lowres_image.png"
    if not he_path.exists():
        raise FileNotFoundError(f"找不到 HE 图像: {slide_dir}/spatial/")

    img_array = load_he_image(he_path)
    h, w      = img_array.shape[:2]
    he_image  = Image.fromarray(img_array)
    half      = patch_size // 2

    patches = {}
    skipped = 0

    for barcode, row in tqdm(obs_df.iterrows(),
                              total=len(obs_df),
                              desc=f"  [{slide_id}] patches",
                              leave=False):
        cx, cy = int(row["pixel_x"]), int(row["pixel_y"])
        x0, y0 = cx - half, cy - half
        x1, y1 = cx + half, cy + half

        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            skipped += 1
            continue

        patch = he_image.crop((x0, y0, x1, y1))
        patch = patch.resize((patch_size, patch_size), Image.Resampling.BICUBIC)
        patches[barcode] = np.array(patch, dtype=np.uint8)

    logger.info(f"  [{slide_id}] {len(patches)} patches, {skipped} skipped")
    return patches


# ─────────────────────────────────────────────
# Step 4: 图像特征提取（UNI2 + CONCH）
# ─────────────────────────────────────────────

def load_encoders(device: str):
    """加载 UNI2 和 CONCH 编码器，返回 (uni_model, uni_tf, conch_model, conch_tf)"""
    import torch.nn as nn
    from timm.models.vision_transformer import VisionTransformer
    from timm.layers.mlp import GluMlp
    from torchvision import transforms

    logger.info(f"加载 UNI2 from {UNI2_WEIGHTS_DIR}")
    ckpt_files = (list(UNI2_WEIGHTS_DIR.glob("*.bin")) +
                  list(UNI2_WEIGHTS_DIR.glob("*.safetensors")) +
                  list(UNI2_WEIGHTS_DIR.glob("*.pth")))
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

    # CONCH
    from conch.open_clip_custom import create_model_from_pretrained
    logger.info(f"加载 CONCH from {CONCH_WEIGHTS_DIR}")
    conch_model, conch_tf = create_model_from_pretrained(
        "conch_ViT-B-16", str(CONCH_WEIGHTS_DIR / "pytorch_model.bin")
    )
    conch_model.eval().to(device)

    return uni_model, uni_tf, conch_model, conch_tf


def extract_features_all(patches: dict, uni_model, uni_tf,
                          conch_model, conch_tf,
                          device: str, batch_size: int = 64) -> dict:
    """
    对所有 patch 提取 UNI2(1536) + CONCH(512) = 2048 维特征。
    Returns: {barcode: np.ndarray (2048,)}
    """
    barcodes = list(patches.keys())
    results  = {}

    uni_ln   = torch.nn.LayerNorm(1536).to(device)
    conch_ln = torch.nn.LayerNorm(512).to(device)

    for i in tqdm(range(0, len(barcodes), batch_size),
                  desc="  特征提取", leave=False):
        batch_bc  = barcodes[i : i + batch_size]
        batch_img = [patches[b] for b in batch_bc]

        uni_t = torch.stack([
            uni_tf(Image.fromarray(img)) for img in batch_img
        ]).to(device)
        with torch.no_grad():
            z_uni = uni_ln(uni_model(uni_t))         # (B, 1536)

        conch_t = torch.stack([
            conch_tf(Image.fromarray(img)) for img in batch_img
        ]).to(device)
        with torch.no_grad():
            z_conch = conch_ln(
                conch_model.encode_image(conch_t, proj_contrast=False)
            )                                         # (B, 512)

        z = torch.cat([z_uni, z_conch], dim=-1).cpu().numpy()  # (B, 2048)
        for j, bc in enumerate(batch_bc):
            results[bc] = z[j]

    return results


# ─────────────────────────────────────────────
# Step 5: 空间邻居索引（每个 slide 单独建）
# ─────────────────────────────────────────────

def build_neighbor_map(adata_all, radius: float = 150.0) -> dict:
    """
    在同一 slide 内建 KD-tree，跨 slide 不建邻居关系。

    Returns: {prefixed_barcode: [prefixed_barcode, ...]}
    """
    logger.info(f"建空间邻居索引 (radius={radius})...")
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
    logger.info(f"  平均邻居数: {n_avg:.2f}")
    return neighbor_map


# ─────────────────────────────────────────────
# Step 6: Leave-one-out splits
# ─────────────────────────────────────────────

def make_splits(adata_all) -> list:
    """每个 slide 轮流作为测试集，其余作为训练集。"""
    slides = sorted(adata_all.obs["slide_id"].unique().tolist())
    splits = []
    for test_slide in slides:
        test_mask  = adata_all.obs["slide_id"] == test_slide
        train_mask = ~test_mask
        splits.append({
            "test_slide": test_slide,
            "train":      adata_all.obs_names[train_mask].tolist(),
            "test":       adata_all.obs_names[test_mask].tolist(),
        })
    logger.info(f"生成 {len(splits)} 个 leave-one-out folds")
    return splits


# ─────────────────────────────────────────────
# Step 7: 保存
# ─────────────────────────────────────────────

def save_all(output_dir: Path, adata_all, z_img_all: dict,
             neighbor_map: dict, splits: list, gene_names: list):

    output_dir.mkdir(parents=True, exist_ok=True)

    barcodes    = adata_all.obs_names.tolist()
    gene_matrix = adata_all.X
    if sp.issparse(gene_matrix):
        gene_matrix = gene_matrix.toarray()
    gene_matrix = gene_matrix.astype(np.float32)

    # 基因表达
    np.save(output_dir / "gene_expression.npy", gene_matrix)
    logger.info(f"gene_expression.npy : {gene_matrix.shape}")

    # 图像特征（按 barcode 顺序对齐，缺失的用零填充）
    missing  = [bc for bc in barcodes if bc not in z_img_all]
    if missing:
        logger.warning(f"  {len(missing)} 个 barcode 没有图像特征，用零填充")
    feat_dim  = next(iter(z_img_all.values())).shape[0]
    z_img_mat = np.stack([
        z_img_all.get(bc, np.zeros(feat_dim)) for bc in barcodes
    ]).astype(np.float32)
    np.save(output_dir / "z_img_features.npy", z_img_mat)
    logger.info(f"z_img_features.npy  : {z_img_mat.shape}")

    # 元数据
    with open(output_dir / "barcodes.json", "w") as f:
        json.dump(barcodes, f)
    with open(output_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)
    with open(output_dir / "neighbor_map.json", "w") as f:
        json.dump(neighbor_map, f)
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    adata_all.obs.to_csv(output_dir / "obs_metadata.csv")

    # bc2slide：供 dataset.py 的 _parse_slide_ids 使用
    bc2slide = adata_all.obs["slide_id"].to_dict()  # {prefixed_bc: slide_id}

    summary = {
        "n_spots":     int(gene_matrix.shape[0]),
        "n_genes":     int(gene_matrix.shape[1]),
        "n_slides":    int(len(splits)),
        "feature_dim": int(z_img_mat.shape[1]),
        "gene_mean":   float(gene_matrix.mean()),
        "gene_std":    float(gene_matrix.std()),
        "bc2slide":    bc2slide,   # dataset.py 用于解析 slide_id
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("预处理完成，summary:")
    for k, v in summary.items():
        if k != "bc2slide":   # bc2slide 太长，不打印
            logger.info(f"  {k}: {v}")
    logger.info(f"输出目录: {output_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device     = args.device if torch.cuda.is_available() else "cpu"

    logger.info(f"Device    : {device}")
    logger.info(f"Data dir  : {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"n_hvg     : {args.n_hvg}")

    # ── 找到所有 slide 目录 ──
    if args.slides:
        slide_dirs = [(data_dir / s, s) for s in args.slides]
    else:
        slide_dirs = sorted([
            (d, d.name) for d in data_dir.iterdir()
            if d.is_dir() and (d / "filtered_feature_bc_matrix.h5").exists()
        ])

    logger.info(f"找到 {len(slide_dirs)} 个 slide: {[s for _, s in slide_dirs]}")

    # ── Step 1: 每个 slide 的基因表达预处理 ──
    logger.info("\n=== Step 1: 基因表达预处理 ===")
    adatas = []
    for slide_dir, slide_id in slide_dirs:
        logger.info(f"处理 slide: {slide_id}")
        try:
            a = load_one_slide_expr(
                slide_dir, slide_id, args.min_genes, args.min_cells
            )
            adatas.append(a)
        except Exception as e:
            logger.error(f"  [{slide_id}] 失败: {e}")

    if not adatas:
        raise RuntimeError("所有 slide 处理失败")

    # ── Step 2: 全局 HVG 对齐 ──
    logger.info("\n=== Step 2: 全局 HVG 筛选 ===")
    adata_all, gene_names = global_hvg(adatas, args.n_hvg)

    # ── Step 3 & 4: patch 提取 + 图像特征 ──
    logger.info("\n=== Step 3 & 4: patch 提取 + 图像特征提取 ===")
    uni_model, uni_tf, conch_model, conch_tf = load_encoders(device)

    z_img_all = {}
    for slide_dir, slide_id in slide_dirs:
        mask   = adata_all.obs["slide_id"] == slide_id
        obs_df = adata_all.obs[mask][["pixel_x", "pixel_y"]]

        if len(obs_df) == 0:
            logger.warning(f"  [{slide_id}] 没有有效 spot，跳过")
            continue

        try:
            patches = extract_patches_for_slide(
                slide_dir, slide_id, obs_df, args.patch_size
            )
            feats = extract_features_all(
                patches, uni_model, uni_tf, conch_model, conch_tf,
                device=device, batch_size=args.batch_size,
            )
            z_img_all.update(feats)
            logger.info(f"  [{slide_id}] 提取特征 {len(feats)} 个")
        except Exception as e:
            logger.error(f"  [{slide_id}] 特征提取失败: {e}")

    del uni_model, conch_model
    torch.cuda.empty_cache()

    # ── Step 5: 空间邻居 ──
    logger.info("\n=== Step 5: 建空间邻居索引 ===")
    neighbor_map = build_neighbor_map(adata_all, radius=args.neighbor_r)

    # ── Step 6: Splits ──
    logger.info("\n=== Step 6: 生成 leave-one-out splits ===")
    splits = make_splits(adata_all)

    # ── Step 7: 保存 ──
    logger.info("\n=== Step 7: 保存 ===")
    save_all(output_dir, adata_all, z_img_all,
             neighbor_map, splits, gene_names)


if __name__ == "__main__":
    main()