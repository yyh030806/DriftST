"""
preprocess_her2st.py
====================
HER2ST (HEST格式) 预处理脚本，基因表达处理完全对齐 STEM：
  - gene_expression.npy 存 RAW counts（STEM 不对 adata_lst 做归一化）
  - 基因列表直接使用 STEM 的 select_gene_list.txt
  - patch 大小从 adata.uns["spatial"] 读真实 spot 直径（与 STEM 一致）
  - 图像编码器升级为 UNI2(1536) + CONCH(512) = 2048 维（DriftST 设定）

输入：st/{SPA119..SPA154}.h5ad + wsis/{SPA119..SPA154}.tif
输出：DriftST 格式（gene_expression.npy / z_img_features.npy / ...）

用法：
    python preprocess_her2st.py \
        --data_dir  /data/buyonggan/DriftST/hest1k_datasets/her2st \
        --output_dir /data/buyonggan/DriftST/hest1k_datasets/her2st/processed_data \
        --gene_list  /data/buyonggan/DriftST/hest1k_datasets/her2st/processed_data/select_gene_list.txt

Bug 修复：
  [Bug4] adatas 从 list 改为 dict（key=slide_id），消除某 slide 加载失败时
         adatas[fn_lst.index(slide_id)] 索引错位、读到错误 spot_diameter 的问题
"""

import os
import json
import logging
import argparse
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

WEIGHTS_DIR       = Path("/data/buyonggan/DriftST/weights")
UNI2_WEIGHTS_DIR  = WEIGHTS_DIR / "uni2"
CONCH_WEIGHTS_DIR = WEIGHTS_DIR / "conch"


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, required=True,
                   help="her2st 根目录，下面有 st/ 和 wsis/ 子目录")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gene_list",  type=str, required=True,
                   help="STEM 的 select_gene_list.txt 路径")
    p.add_argument("--neighbor_r", type=float, default=150.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--slides",     type=str, nargs="*", default=None,
                   help="指定处理哪些 slide，默认全部 SPA119-SPA154")
    return p.parse_args()


# ─────────────────────────────────────────────
# Step 1: 加载基因列表（直接用 STEM 的）
# ─────────────────────────────────────────────

def load_gene_list(gene_list_path: str):
    with open(gene_list_path, "r") as f:
        genes = [l.strip() for l in f if l.strip()]
    logger.info(f"加载基因列表: {len(genes)} 个基因 ({gene_list_path})")
    return genes


# ─────────────────────────────────────────────
# Step 2: 加载单个 slide（HEST h5ad 格式）
# ─────────────────────────────────────────────

def load_one_slide(st_path: Path, slide_id: str, gene_list: list):
    """
    加载 HEST 格式的 h5ad，存 RAW counts，完全对齐 STEM。

    STEM 的 adata_lst 始终保持 raw counts（只有用于 HVG 统计的 copy 才归一化），
    all_count_df 直接从原始 adata_lst 取值，因此 gene_expression.npy 必须存 raw。
    """
    h5ad_file = st_path / f"{slide_id}.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(f"找不到: {h5ad_file}")

    adata = sc.read_h5ad(str(h5ad_file))

    # ── 取公共基因子集（对齐 STEM 的 common_genes 步骤）──
    # 缺失基因补0，保证所有 slide 维度一致
    missing = [g for g in gene_list if g not in adata.var_names]
    if missing:
        logger.warning(f"  [{slide_id}] {len(missing)} 个基因不存在，补0列")

    available = [g for g in gene_list if g in adata.var_names]
    adata_sub = adata[:, available].copy()   # ← RAW counts，不做任何归一化

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

    # ── spatial 坐标 ──
    if "spatial" not in adata.obsm:
        raise ValueError(f"[{slide_id}] 缺少 obsm['spatial']")
    adata_sub.obsm["spatial"] = adata.obsm["spatial"]

    # ── 读取真实 spot 直径（对齐 STEM 的 patch 大小逻辑）──
    spot_diameter = None
    spatial_meta  = adata.uns.get("spatial", {})
    if isinstance(spatial_meta, dict):
        for key, val in spatial_meta.items():
            if isinstance(val, dict) and "scalefactors" in val:
                spot_diameter = val["scalefactors"].get("spot_diameter_fullres")
                break
    adata_sub.uns["spot_diameter_fullres"] = float(spot_diameter) if spot_diameter else None

    # obs 元数据
    adata_sub.obs["slide_id"] = slide_id
    adata_sub.obs["pixel_x"]  = adata_sub.obsm["spatial"][:, 0]
    adata_sub.obs["pixel_y"]  = adata_sub.obsm["spatial"][:, 1]
    adata_sub.obs_names       = [f"{slide_id}_{bc}" for bc in adata_sub.obs_names]

    logger.info(f"  [{slide_id}] {adata_sub.n_obs} spots x {adata_sub.n_vars} genes"
                f"  spot_diameter={spot_diameter}")
    return adata_sub


# ─────────────────────────────────────────────
# Step 3: 加载编码器
# ─────────────────────────────────────────────

def load_encoders(device: str):
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

    from conch.open_clip_custom import create_model_from_pretrained
    logger.info(f"加载 CONCH from {CONCH_WEIGHTS_DIR}")
    conch_model, conch_tf = create_model_from_pretrained(
        "conch_ViT-B-16", str(CONCH_WEIGHTS_DIR / "pytorch_model.bin")
    )
    conch_model.eval().to(device)

    return uni_model, uni_tf, conch_model, conch_tf


# ─────────────────────────────────────────────
# Step 4: patch 提取 + 特征编码
# ─────────────────────────────────────────────

def extract_features_for_slide(wsi_path: Path, slide_id: str,
                                obs_df: pd.DataFrame,
                                spot_diameter: float,        # 真实 spot 直径
                                uni_model, uni_tf, conch_model, conch_tf,
                                device: str, batch_size: int) -> dict:
    """
    返回 {prefixed_barcode: np.ndarray (2048,)}

    patch 大小对齐 STEM：
        radius = max(112, int(spot_diameter // 2))
        patch  = img.crop((x-radius, y-radius, x+radius, y+radius))
    模型输入统一 resize 到 224（UNI2）/ 256（CONCH）。
    """
    tif_file = wsi_path / f"{slide_id}.tif"
    if not tif_file.exists():
        raise FileNotFoundError(f"找不到 WSI: {tif_file}")

    # ── STEM 的 radius 逻辑 ──
    radius = max(112, int(spot_diameter // 2)) if spot_diameter else 112
    logger.info(f"  [{slide_id}] spot_diameter={spot_diameter:.1f}  radius={radius}")

    image = Image.open(str(tif_file)).convert("RGB")
    w, h  = image.size

    uni_ln   = nn.LayerNorm(1536).to(device)
    conch_ln = nn.LayerNorm(512).to(device)

    barcodes = obs_df.index.tolist()
    results  = {}
    skipped  = 0

    for i in tqdm(range(0, len(barcodes), batch_size),
                  desc=f"  [{slide_id}] 特征提取", leave=False):
        batch_bc = barcodes[i : i + batch_size]
        patches, valid_bc = [], []

        for bc in batch_bc:
            cx = float(obs_df.loc[bc, "pixel_x"])
            cy = float(obs_df.loc[bc, "pixel_y"])
            x0 = cx - radius
            y0 = cy - radius
            x1 = cx + radius
            y1 = cy + radius
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                skipped += 1
                continue
            # 裁剪原始大小 patch（与 STEM 一致），模型内部 resize
            patch = image.crop((x0, y0, x1, y1))
            patches.append(patch)
            valid_bc.append(bc)

        if not patches:
            continue

        with torch.no_grad():
            # UNI2：resize 到 224
            uni_t = torch.stack([
                uni_tf(p.resize((224, 224), Image.Resampling.LANCZOS))
                for p in patches
            ]).to(device)
            z_uni = uni_ln(uni_model(uni_t))                       # (B, 1536)

            # CONCH：resize 到 256（对齐 STEM 的 base_width=256）
            conch_t = torch.stack([
                conch_tf(p.resize((256, 256), Image.Resampling.LANCZOS))
                for p in patches
            ]).to(device)
            z_conch = conch_ln(
                conch_model.encode_image(conch_t,
                                         proj_contrast=False,
                                         normalize=False)
            )                                                       # (B, 512)

        z = torch.cat([z_uni, z_conch], dim=-1).cpu().numpy()      # (B, 2048)
        for j, bc in enumerate(valid_bc):
            results[bc] = z[j]

    image.close()
    logger.info(f"  [{slide_id}] 提取 {len(results)} 个，跳过边缘 {skipped} 个")
    return results


# ─────────────────────────────────────────────
# Step 5: 空间邻居
# ─────────────────────────────────────────────

def build_neighbor_map(adata_all, radius: float) -> dict:
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
# Step 6: 保存
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
        logger.warning(f"  {len(missing)} 个 barcode 没有图像特征，用零填充")
    feat_dim  = next(iter(z_img_all.values())).shape[0]
    z_img_mat = np.stack([
        z_img_all.get(bc, np.zeros(feat_dim)) for bc in barcodes
    ]).astype(np.float32)
    np.save(output_dir / "z_img_features.npy", z_img_mat)
    logger.info(f"z_img_features.npy  : {z_img_mat.shape}")

    with open(output_dir / "barcodes.json",    "w") as f: json.dump(barcodes,    f)
    with open(output_dir / "gene_names.json",  "w") as f: json.dump(gene_names,  f)
    with open(output_dir / "neighbor_map.json","w") as f: json.dump(neighbor_map,f)
    with open(output_dir / "splits.json",      "w") as f: json.dump(splits, f, indent=2)

    adata_all.obs.to_csv(output_dir / "obs_metadata.csv")

    bc2slide = adata_all.obs["slide_id"].to_dict()
    summary  = {
        "n_spots":     int(gene_matrix.shape[0]),
        "n_genes":     int(gene_matrix.shape[1]),
        "n_slides":    int(len(splits)),
        "feature_dim": int(z_img_mat.shape[1]),
        "gene_mean":   float(gene_matrix.mean()),
        "gene_std":    float(gene_matrix.std()),
        "bc2slide":    bc2slide,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("预处理完成，summary:")
    for k, v in summary.items():
        if k != "bc2slide":
            logger.info(f"  {k}: {v}")
    logger.info(f"输出目录: {output_dir}")


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

    logger.info(f"Device     : {device}")
    logger.info(f"Data dir   : {data_dir}")
    logger.info(f"Output dir : {output_dir}")

    # slide 列表
    if args.slides:
        fn_lst = args.slides
    else:
        fn_lst = [f"SPA{i}" for i in range(119, 155)]
    logger.info(f"共 {len(fn_lst)} 个 slide: {fn_lst}")

    # ── Step 1: 基因列表 ──
    gene_list = load_gene_list(args.gene_list)

    # ── Step 2: 加载所有 slide ──
    # [Bug4 fix] adatas 改为 dict（key=slide_id），消除 list 索引错位问题。
    # 原来用 list 存储，当某个 slide 加载失败被 try/except 跳过后，
    # adatas[fn_lst.index(slide_id)] 会读到错误 slide 的 spot_diameter。
    # 改为 dict 后，始终通过 adatas_dict[slide_id] 精确取值。
    logger.info("\n=== Step 2: 加载 slide 基因表达 ===")
    adatas_dict: dict = {}                        # [Bug4 fix] list → dict
    for slide_id in fn_lst:
        try:
            adatas_dict[slide_id] = load_one_slide(st_path, slide_id, gene_list)
        except Exception as e:
            logger.error(f"  [{slide_id}] 失败: {e}")

    if not adatas_dict:
        raise RuntimeError("所有 slide 加载失败")

    # [Bug4 fix] 从 dict 取 values 合并，顺序与 fn_lst 保持一致
    adata_all = anndata.concat(
        [adatas_dict[sid] for sid in fn_lst if sid in adatas_dict],
        axis=0, merge="same"
    )
    adata_all.var_names_make_unique()
    logger.info(f"合并后: {adata_all.shape[0]} spots x {adata_all.shape[1]} genes")

    # ── Step 3: 加载编码器 ──
    logger.info("\n=== Step 3: 加载编码器 ===")
    uni_model, uni_tf, conch_model, conch_tf = load_encoders(device)

    # ── Step 4: 特征提取 ──
    logger.info("\n=== Step 4: 图像特征提取 ===")
    z_img_all = {}
    for slide_id in fn_lst:
        if slide_id not in adatas_dict:
            logger.warning(f"  [{slide_id}] Step 2 加载失败，跳过特征提取")
            continue

        mask   = adata_all.obs["slide_id"] == slide_id
        obs_df = adata_all.obs[mask][["pixel_x", "pixel_y"]]
        if len(obs_df) == 0:
            logger.warning(f"  [{slide_id}] 无有效 spot，跳过")
            continue

        # [Bug4 fix] 通过 dict 精确取该 slide 的 spot_diameter
        # 原来用 adatas[fn_lst.index(slide_id)] 取，slide 加载失败时索引错位
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
            logger.error(f"  [{slide_id}] 特征提取失败: {e}")

    del uni_model, conch_model
    torch.cuda.empty_cache()

    # ── Step 5: 空间邻居 ──
    logger.info("\n=== Step 5: 建空间邻居索引 ===")
    neighbor_map = build_neighbor_map(adata_all, radius=args.neighbor_r)

    # ── Step 6: Leave-one-out splits ──
    logger.info("\n=== Step 6: 生成 leave-one-out splits ===")
    slides = sorted(adata_all.obs["slide_id"].unique().tolist())
    splits = []
    for test_slide in slides:
        test_mask  = adata_all.obs["slide_id"] == test_slide
        splits.append({
            "test_slide": test_slide,
            "train":      adata_all.obs_names[~test_mask].tolist(),
            "test":       adata_all.obs_names[test_mask].tolist(),
        })
    logger.info(f"生成 {len(splits)} 个 fold")

    # ── Step 7: 保存 ──
    logger.info("\n=== Step 7: 保存 ===")
    save_all(output_dir, adata_all, z_img_all, neighbor_map, splits, gene_list)


if __name__ == "__main__":
    main()