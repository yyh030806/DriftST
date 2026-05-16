"""加载 best ckpt 在 val 集上跑 evaluate。

用法（参数与 train.py 保持一致，必须传与训练相同的模型架构超参）:
  python test.py \
      --data_dir hest1k_datasets/xenium_janesick/processed_data \
      --fold 0 \
      --ckpt experiments/xenium_xxxx/fold_0/best.pt \
      --use_neighbor --use_gate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import build_datasets
from src.model import GenePredictor
from src.evaluation import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--fold",         type=int, required=True)
    p.add_argument("--ckpt",         type=str, required=True,
                   help="best.pt 路径")
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--batch_size",   type=int, default=256)

    # ─── 必须与训练一致的模型结构超参 ───
    p.add_argument("--input_dim",    type=int,   default=2048)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--num_layers",   type=int,   default=4)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--n_attn_layers", type=int,  default=2)
    p.add_argument("--use_gate",     action="store_true")
    p.add_argument("--gate_targets", type=float, nargs="+", default=None)
    p.add_argument("--gate_entropy_weight", type=float, default=0.1)
    p.add_argument("--use_neighbor", action="store_true")
    p.add_argument("--max_neighbors", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── SVG ranking（用于 SVG-PCC）───────────────────────────
    svg_indices = None
    svg_path = Path(args.data_dir) / "svg_ranking.json"
    if svg_path.exists():
        svg_data = json.load(open(svg_path))
        gene_names = json.load(open(Path(args.data_dir) / "gene_names.json"))
        gene2idx = {g: i for i, g in enumerate(gene_names)}
        svg_indices = [gene2idx[g] for g in svg_data["ranking"] if g in gene2idx]
        print(f"SVG ranking 加载完成，共 {len(svg_indices)} 个基因")

    # ── 数据 ─────────────────────────────────────────────────
    raw_train_ds, val_ds, meta = build_datasets(
        data_dir=args.data_dir, fold=args.fold
    )
    n_genes = meta["n_genes"]
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 基因顺序（与训练一致：层次聚类得到）─────────────────
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    print("重建基因聚类排列（与训练时算法一致）...")
    exprs = raw_train_ds.gene_expr.numpy()
    corr_dist = pdist(exprs.T, metric="correlation")
    corr_dist = np.nan_to_num(corr_dist, nan=2.0, posinf=2.0, neginf=0.0)
    Z = linkage(corr_dist, method="ward")
    gene_order = leaves_list(Z).tolist()

    # ── 模型 ─────────────────────────────────────────────────
    model = GenePredictor(
        input_dim    = args.input_dim,
        hidden_dim   = args.hidden_dim,
        num_layers   = args.num_layers,
        num_heads    = args.num_heads,
        output_dim   = n_genes,
        dropout      = args.dropout,
        n_attn_layers = args.n_attn_layers,
        gene_order   = gene_order,
        use_gate     = args.use_gate,
        gate_target_fractions = args.gate_targets,
        gate_entropy_weight   = args.gate_entropy_weight,
        use_neighbor = args.use_neighbor,
        max_neighbors = args.max_neighbors,
    ).to(device)

    # 训练时 load 的共表达矩阵需要重建（结构里有 bio_bias buffer）
    R = np.corrcoef(exprs.T)
    R = np.nan_to_num(R, nan=0.0)
    model.load_bio_bias(R)

    # ── 加载权重 ─────────────────────────────────────────────
    print(f"加载 ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # ── 评估 ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nfold {args.fold} 评估\n{'='*60}")
    evaluate(model, val_loader, n_genes, device, svg_indices=svg_indices)


if __name__ == "__main__":
    main()
