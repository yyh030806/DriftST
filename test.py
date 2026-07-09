"""Evaluate a trained DriftST checkpoint on a validation/test split.

Usage:
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
from src.evaluation import evaluate, metrics_from_arrays
from src.postprocess import gene_stats, variance_postprocess


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--fold",         type=int, required=True)
    p.add_argument("--ckpt",         type=str, required=True,
                   help="path to a trained checkpoint")
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--save_h5ad",    type=str, default=None,
                   help="export AnnData with X=pred, layers['gt']=gt, obsm['spatial']=coordinates")
    p.add_argument("--variance-postproc", "--variance_postproc",
                   dest="variance_postproc",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="enable per-gene affine variance calibration against the training split")
    p.add_argument("--postproc_alpha", type=float, default=1.2,
                   help="variance multiplier; 1.0 matches the reference variance")

    # Model hyperparameters must match training.
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

    svg_indices = None
    svg_path = Path(args.data_dir) / "svg_ranking.json"
    if svg_path.exists():
        svg_data = json.load(open(svg_path))
        gene_names = json.load(open(Path(args.data_dir) / "gene_names.json"))
        gene2idx = {g: i for i, g in enumerate(gene_names)}
        svg_indices = [gene2idx[g] for g in svg_data["ranking"] if g in gene2idx]
        print(f"Loaded SVG ranking with {len(svg_indices)} genes")

    raw_train_ds, val_ds, meta = build_datasets(
        data_dir=args.data_dir, fold=args.fold, max_neighbors=args.max_neighbors
    )
    n_genes = meta["n_genes"]
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = GenePredictor(
        input_dim    = args.input_dim,
        hidden_dim   = args.hidden_dim,
        num_layers   = args.num_layers,
        num_heads    = args.num_heads,
        output_dim   = n_genes,
        dropout      = args.dropout,
        n_attn_layers = args.n_attn_layers,
        use_gate     = args.use_gate,
        gate_target_fractions = args.gate_targets,
        gate_entropy_weight   = args.gate_entropy_weight,
        use_neighbor = args.use_neighbor,
        max_neighbors = args.max_neighbors,
    ).to(device)

    # Reconstruct the training co-expression matrix used by the bio_bias buffer.
    exprs = raw_train_ds.gene_expr.numpy()
    R = np.corrcoef(exprs.T)
    R = np.nan_to_num(R, nan=0.0)
    model.load_bio_bias(R)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Legacy checkpoints may carry a gene_order permutation.
    legacy_order = ckpt.get("gene_order")
    if legacy_order is not None and legacy_order != list(range(n_genes)):
        legacy_inv = torch.argsort(torch.tensor(legacy_order, dtype=torch.long)).to(device)
        original_forward = model.forward
        def legacy_forward(*a, **kw):
            x0, info = original_forward(*a, **kw)
            return x0[:, legacy_inv], info
        model.forward = legacy_forward
        print("[legacy] Applied output permutation from checkpoint gene_order")

    print(f"\n{'='*60}\nfold {args.fold} evaluation\n{'='*60}")
    need_pred = bool(args.save_h5ad) or args.variance_postproc
    results = evaluate(model, val_loader, n_genes, device,
                       svg_indices=svg_indices,
                       return_predictions=need_pred)

    if args.variance_postproc:
        all_pred, all_true = results[-2], results[-1]
        ref_mean, ref_std = gene_stats(exprs)
        all_pred_cal = variance_postprocess(
            all_pred, ref_mean, ref_std, alpha=args.postproc_alpha)
        std_ratio = (all_pred_cal.std(0) / (all_true.std(0) + 1e-12)).mean()
        print(f"\n[variance calibration] alpha={args.postproc_alpha}  "
              f"std(pred)/std(gt) {all_pred.std(0).mean()/ (all_true.std(0).mean()+1e-12):.3f} "
              f"-> {std_ratio:.3f}")
        print("[calibrated metrics]")
        metrics_from_arrays(all_pred_cal, all_true, n_genes, svg_indices=svg_indices)
        results = (*results[:-2], all_pred_cal, all_true)

    if args.save_h5ad:
        import anndata as ad
        import pandas as pd
        all_pred, all_true = results[-2], results[-1]

        gene_names = json.load(open(Path(args.data_dir) / "gene_names.json"))[:n_genes]
        barcodes   = val_ds.barcodes
        obs_meta   = pd.read_csv(Path(args.data_dir) / "obs_metadata.csv", index_col=0)
        obs_sub    = obs_meta.loc[barcodes].copy()

        xcol = "pixel_x" if "pixel_x" in obs_sub.columns else "x"
        ycol = "pixel_y" if "pixel_y" in obs_sub.columns else "y"
        spatial = obs_sub[[xcol, ycol]].to_numpy()

        adata = ad.AnnData(
            X     = all_pred.astype("float32"),
            obs   = obs_sub,
            var   = pd.DataFrame(index=gene_names),
            obsm  = {"spatial": spatial},
            layers= {"gt": all_true.astype("float32")},
        )
        adata.uns["fold"] = args.fold
        if args.variance_postproc:
            adata.uns["note"] = ("X = predicted ln(x+1), variance post-processed "
                                 f"(per-gene affine calib to train dist, alpha={args.postproc_alpha}); "
                                 "layers['gt'] = ground truth ln(x+1)")
        else:
            adata.uns["note"] = "X = predicted ln(x+1) expression; layers['gt'] = ground truth ln(x+1)"

        out_path = Path(args.save_h5ad)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_path)
        print(f"Exported {out_path}  shape={adata.shape}")


if __name__ == "__main__":
    main()
