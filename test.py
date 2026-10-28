"""Evaluate a trained DriftST checkpoint on the held-out test split.

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
from src.evaluation import evaluate, load_svg_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--fold",         type=int, required=True)
    p.add_argument("--ckpt",         type=str, required=True,
                   help="path to a trained checkpoint")
    p.add_argument("--splits_file",  type=str, default=None,
                   help="explicit split file; must match training")
    p.add_argument("--svg_ranking",  type=str, default=None,
                   help="training-only SVG ranking for this fold")
    p.add_argument("--allow_legacy_checkpoint", action="store_true",
                   help="allow checkpoints lacking split provenance (results may be biased)")
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--save_h5ad",    type=str, default=None,
                   help="export AnnData with X=pred, layers['gt']=gt, obsm['spatial']=coordinates")

    # Current checkpoints supply these settings; explicit overrides must match.
    p.add_argument("--input_dim",    type=int, default=None)
    p.add_argument("--hidden_dim",   type=int, default=None)
    p.add_argument("--num_layers",   type=int, default=None)
    p.add_argument("--num_heads",    type=int, default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--n_attn_layers", type=int, default=None)
    p.add_argument("--use_gate",     action="store_true", default=None)
    p.add_argument("--gate_targets", type=float, nargs="+", default=None)
    p.add_argument("--gate_entropy_weight", type=float, default=None)
    p.add_argument("--use_neighbor", action="store_true", default=None)
    p.add_argument("--max_neighbors", type=int, default=None)
    return p.parse_args()


def resolve_model_settings(args, ckpt, legacy):
    defaults = {"input_dim": 2048, "hidden_dim": 256, "num_layers": 4, "num_heads": 8,
                "dropout": 0.1, "n_attn_layers": 2, "use_gate": False, "gate_targets": None,
                "gate_entropy_weight": 0.1, "use_neighbor": False, "max_neighbors": 6}
    saved = ckpt.get("args", {})
    if not isinstance(saved, dict):
        saved = {}
    for key, default in defaults.items():
        requested = getattr(args, key)
        if not legacy:
            if key not in saved:
                raise ValueError(f"checkpoint is missing model setting {key}")
            if requested is not None and requested != saved[key]:
                raise ValueError(f"{key} does not match the checkpoint's training setting")
            value = saved[key]
        else:
            value = requested if requested is not None else saved.get(key, default)
        setattr(args, key, value)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    checkpoint_split = ckpt.get("split_sha256")
    legacy = checkpoint_split is None
    if legacy:
        if not args.allow_legacy_checkpoint:
            raise ValueError("checkpoint has no split provenance; legacy checkpoints may have used test data "
                             "for model selection. Pass --allow_legacy_checkpoint to inspect it explicitly")
        print("[warn] legacy checkpoint lacks split provenance; its test results may be biased")
    resolve_model_settings(args, ckpt, legacy)

    raw_train_ds, _, test_ds, meta = build_datasets(
        data_dir=args.data_dir, fold=args.fold, max_neighbors=args.max_neighbors,
        splits_file=args.splits_file, load_val=False,
    )
    n_genes = meta["n_genes"]
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if not legacy and (checkpoint_split != meta["split_sha256"] or ckpt.get("fold") != args.fold):
        raise ValueError("checkpoint split or fold does not match the requested test split")
    if not legacy and ckpt.get("data_sha256") != meta["data_sha256"]:
        raise ValueError("checkpoint processed data does not match the requested dataset")
    print(f"Checkpoint selection: {ckpt.get('selection_metric', 'legacy/unknown')}")
    ranking_path = args.svg_ranking
    checkpoint_ranking = Path(args.ckpt).parent / "svg_ranking.json"
    if ranking_path is None and checkpoint_ranking.exists():
        ranking_path = checkpoint_ranking
    svg_indices = load_svg_indices(args.data_dir, args.fold, meta["train_sha256"], ranking_path,
                                   data_sha256=meta["data_sha256"])

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

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=not legacy)
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

    print(f"\n{'='*60}\nfold {args.fold} held-out test evaluation\n{'='*60}")
    need_pred = bool(args.save_h5ad)
    results = evaluate(model, test_loader, n_genes, device,
                       svg_indices=svg_indices,
                       return_predictions=need_pred,
                       clip_nonnegative=True)

    if args.save_h5ad:
        import anndata as ad
        import pandas as pd
        all_pred, all_true = results[-2], results[-1]

        gene_names = json.load(open(Path(args.data_dir) / "gene_names.json"))[:n_genes]
        barcodes   = test_ds.barcodes
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
        adata.uns["note"] = (
            "X = model predictions in ln(x+1) space, clipped at 0; "
            "layers['gt'] = ground truth ln(x+1)"
        )

        out_path = Path(args.save_h5ad)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_path)
        print(f"Exported {out_path}  shape={adata.shape}")


if __name__ == "__main__":
    main()
