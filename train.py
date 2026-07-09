"""Training entry point for DriftST.

The training objective combines a ZINB reconstruction term with a one-step
drifting loss. Stochastic dropout predictions are stored in a ring buffer and
used as negative samples for the drift step; each spot's matched expression
profile is used as the positive sample.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import build_datasets
from src.drift_step import PredictionBank, multi_scale_drift_step, drift_loss_fn
from src.model import GenePredictor, zinb_loss
from src.evaluation import evaluate


class NullWandbRun:
    def __init__(self):
        self.summary = {}


class NullWandb:
    """Small no-op logger with the subset of wandb used by this script."""

    def __init__(self):
        self.run = NullWandbRun()
        self.config = self

    def init(self, *args, **kwargs):
        return self.run

    def update(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def finish(self):
        return None


# -----------------------------------------------------------------------------
# Bank warmup
# -----------------------------------------------------------------------------

@torch.no_grad()
def capture_stochastic(model, loader, device, n_genes, K):
    """Collect K dropout samples per spot for distribution visualization."""
    model.train()  # enable dropout
    preds, trues = [], []
    for batch in loader:
        img      = batch["z_img"].to(device)
        nb_img   = batch["neighbor_zimg"].to(device)
        nb_valid = batch["neighbor_valid"].to(device)
        ks = []
        for _ in range(K):
            x0, _ = model(img, nb_img, nb_valid)
            ks.append(x0.cpu().numpy())
        preds.append(np.stack(ks, axis=1).reshape(-1, n_genes))  # (B*K, G)
        trues.append(batch["gene_expr"].numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


@torch.no_grad()
def warmup_bank(model, loader, bank, device, K: int, min_samples: int):
    """Fill the prediction bank with K dropout forwards per spot."""
    model.train()
    collected = 0
    for batch in loader:
        img      = batch["z_img"].to(device)
        nb_img   = batch["neighbor_zimg"].to(device)
        nb_valid = batch["neighbor_valid"].to(device)

        preds = []
        for _ in range(K):
            x0, _ = model(img, nb_img, nb_valid)
            preds.append(x0)
        x0_k = torch.stack(preds, dim=1)                         # (B, K, D)
        bank.enqueue(x0_k.reshape(-1, x0_k.shape[-1]))          # (B*K, D)

        collected += img.shape[0] * K
        if collected >= min_samples:
            break

    print(f"Bank warmup complete: collected {collected} samples "
          f"(bank.count={bank.total_count}/{bank.size})")


# -----------------------------------------------------------------------------
# Memory-efficient K-sample dropout prediction
# -----------------------------------------------------------------------------

def sample_k_predictions(model, img, K: int, nb_img=None, nb_valid=None):
    """Return K stochastic predictions, keeping gradients only for the first."""
    preds = []

    x0, gate_info = model(img, nb_img, nb_valid)
    preds.append(x0.unsqueeze(1))

    with torch.no_grad():
        for _ in range(K - 1):
            x0, _ = model(img, nb_img, nb_valid)
            preds.append(x0.unsqueeze(1))

    return torch.cat(preds, dim=1), gate_info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",         type=str,   required=True)
    p.add_argument("--fold",             type=int,   required=True)
    p.add_argument("--output_dir",       type=str,   required=True)
    p.add_argument("--device",           type=str,   default="cuda")
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--n_genes",          type=int,   default=300)
    p.add_argument("--input_dim",        type=int,   default=2048)
    p.add_argument("--hidden_dim",       type=int,   default=256)
    p.add_argument("--num_layers",       type=int,   default=4)
    p.add_argument("--num_heads",        type=int,   default=8)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--n_attn_layers",   type=int,   default=2,
                   help="number of bio-guided attention layers")
    p.add_argument("--use_gate",        action="store_true",
                   help="enable progressive gene gating")
    p.add_argument("--gate_weight",     type=float, default=0.1,
                   help="weight of the gate sparsity loss")
    p.add_argument("--gate_targets",    type=float, nargs="+", default=None,
                   help="target keep fractions per layer, e.g. 0.8 0.5")
    p.add_argument("--gate_entropy_weight", type=float, default=0.1,
                   help="entropy regularization weight for sharper gates")
    p.add_argument("--std_weight",      type=float, default=0.0,
                   help="weight of per-gene std-matching loss; disabled by default")
    p.add_argument("--balance_slides",  action="store_true",
                   help="balance sampling across training slides")

    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--t_max",            type=int,   default=None)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--wd",               type=float, default=1e-4)
    p.add_argument("--warm_epochs",      type=int,   default=10)
    p.add_argument("--drift_weight",     type=float, default=0.15)
    p.add_argument("--gen_per_spot",     type=int,   default=16)

    p.add_argument("--R_list",           type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--drift_step",       type=float, default=1.0)
    p.add_argument("--bank_size",        type=int,   default=8192)
    p.add_argument("--bank_sample_size", type=int,   default=256)
    p.add_argument("--use_neighbor",    action="store_true",
                   help="enable spatial neighbor aggregation")
    p.add_argument("--max_neighbors",   type=int,   default=6)

    p.add_argument("--snapshot_epochs", type=int, nargs="+", default=None,
                   help="epochs at which validation prediction snapshots are saved")
    p.add_argument("--snapshot_dir",    type=str, default=None,
                   help="directory for snapshot npz files")

    p.add_argument("--wandb_project",    type=str,   default="DriftST")
    p.add_argument("--wandb_name",       type=str,   default=None,
                   help="run name; defaults to fold-{fold}")
    p.add_argument("--wandb_offline",    action="store_true",
                   help="use wandb offline mode")
    p.add_argument("--no_wandb",         action="store_true",
                   help="disable wandb logging")

    args = p.parse_args()

    if args.no_wandb:
        wandb = NullWandb()
    else:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        try:
            import wandb
        except ImportError:
            print("[warn] wandb is not installed; logging is disabled.")
            wandb = NullWandb()

    run_name = args.wandb_name or f"fold-{args.fold}"
    wandb.init(
        project = args.wandb_project,
        name    = run_name,
        config  = vars(args),
    )

    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_indices = None
    svg_path = Path(args.data_dir) / "svg_ranking.json"
    if svg_path.exists():
        with open(svg_path) as f:
            svg_data = json.load(f)
        with open(Path(args.data_dir) / "gene_names.json") as f:
            gene_names = json.load(f)
        gene2idx = {g: i for i, g in enumerate(gene_names)}
        svg_indices = [gene2idx[g] for g in svg_data["ranking"] if g in gene2idx]
        print(f"Loaded SVG ranking with {len(svg_indices)} genes from {svg_path.name}")
    else:
        print("svg_ranking.json not found; SVG PCC metrics will be skipped")

    raw_train_ds, val_ds, meta = build_datasets(
        data_dir=args.data_dir, fold=args.fold, max_neighbors=args.max_neighbors
    )

    data_n_genes = meta["n_genes"]
    if args.n_genes != data_n_genes:
        raise ValueError(
            f"[n_genes mismatch] --n_genes={args.n_genes}, but the data contain "
            f"{data_n_genes} genes. Please set --n_genes to {data_n_genes}."
        )
    n_genes    = data_n_genes
    test_slide = meta["test_slide"]

    print(f"Dataset: n_genes={n_genes}, test_slide={test_slide}, "
          f"n_train={meta['n_train']}, n_test={meta['n_test']}")

    if args.balance_slides:
        from collections import Counter
        slide_counts = Counter(raw_train_ds.slide_ids)
        sample_weights = [1.0 / slide_counts[s] for s in raw_train_ds.slide_ids]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(raw_train_ds), replacement=True,
        )
        print(f"Using slide-balanced sampling across {len(slide_counts)} slides; "
              f"spot count min={min(slide_counts.values())}, max={max(slide_counts.values())}")
        train_loader = DataLoader(
            raw_train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            raw_train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers,
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

    print("Computing the gene co-expression matrix (Pearson correlation)...")
    train_expr = (raw_train_ds.gene_expr.numpy()
                  if hasattr(raw_train_ds.gene_expr, "numpy")
                  else np.asarray(raw_train_ds.gene_expr))
    R = np.corrcoef(train_expr.T)
    R = np.nan_to_num(R, nan=0.0)
    model.load_bio_bias(R)
    print(f"Loaded co-expression matrix, shape={R.shape}, "
          f"mean={R.mean():.4f}, nonzero={np.count_nonzero(np.abs(R) > 0.3)}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")
    wandb.config.update({"total_params_M": total_params / 1e6}, allow_val_change=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    t_max = args.t_max if args.t_max is not None else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=1e-5,
    )

    bank = PredictionBank(
        size     = args.bank_size,
        feat_dim = n_genes,
    )
    warmup_bank(
        model, train_loader, bank, device,
        K           = args.gen_per_spot,
        min_samples = args.bank_sample_size * 4,
    )

    best_pcc = -1.0
    best_loss = float("inf")

    def std_match_loss(pred, gt):
        """MSE between per-gene batch standard deviations."""
        return ((pred.std(dim=0) - gt.std(dim=0)) ** 2).mean()

    for epoch in range(args.epochs):
        model.train()
        train_loss        = 0.0
        drift_loss_acc    = 0.0
        zinb_loss_acc     = 0.0
        gate_loss_acc     = 0.0
        std_loss_acc      = 0.0
        is_warmup = (epoch < args.warm_epochs)

        if epoch == args.warm_epochs:
            best_pcc = -1.0
            best_loss = float("inf")
            print("Starting the drift phase")

        for batch in train_loader:
            img      = batch["z_img"].to(device)
            g_true   = batch["gene_expr"].to(device)
            g_counts = batch["gene_counts"].to(device)
            nb_img   = batch["neighbor_zimg"].to(device)
            nb_valid = batch["neighbor_valid"].to(device)
            B        = img.shape[0]

            if is_warmup:
                x0, gate_info = model(img, nb_img, nb_valid)
                loss = zinb_loss(g_counts, x0,
                                 gate_info["zinb_theta"], gate_info["zinb_pi_logits"])

                if args.std_weight > 0:
                    s_loss = std_match_loss(x0, g_true)
                    loss   = loss + args.std_weight * s_loss
                    std_loss_acc += s_loss.item()

                if gate_info is not None:
                    g_loss = gate_info.get('gate_sparsity_loss')
                    if g_loss is not None:
                        loss = loss + args.gate_weight * g_loss
                        gate_loss_acc += g_loss.item()

                # Keep the bank populated during warmup.
                with torch.no_grad():
                    preds = [model(img, nb_img, nb_valid)[0] for _ in range(args.gen_per_spot)]
                    x0_k_w = torch.stack(preds, dim=1)
                    bank.enqueue(x0_k_w.reshape(-1, n_genes))

            else:
                K    = args.gen_per_spot
                x0_k, gate_info = sample_k_predictions(model, img, K, nb_img, nb_valid)  # (B, K, n_genes)

                recon_loss = zinb_loss(g_counts, x0_k[:, 0, :],
                                       gate_info["zinb_theta"], gate_info["zinb_pi_logits"])

                pos = g_true.detach().unsqueeze(1)

                neg_bank = bank.sample(args.bank_sample_size, device)
                neg_bank = neg_bank.unsqueeze(0).expand(B, -1, -1)

                goal, goal_scaled, scale_inp = multi_scale_drift_step(
                    x      = x0_k,
                    pos    = pos,
                    neg    = neg_bank,
                    R_list = tuple(args.R_list),
                    step   = args.drift_step,
                )

                d_loss = drift_loss_fn(x0_k, goal_scaled, scale_inp)
                loss   = args.drift_weight * d_loss + (1 - args.drift_weight) * recon_loss

                if args.std_weight > 0:
                    s_loss = std_match_loss(x0_k[:, 0, :], g_true)
                    loss   = loss + args.std_weight * s_loss
                    std_loss_acc += s_loss.item()

                if gate_info is not None:
                    g_loss = gate_info.get('gate_sparsity_loss')
                    if g_loss is not None:
                        loss = loss + args.gate_weight * g_loss
                        gate_loss_acc += g_loss.item()

                drift_loss_acc += d_loss.item()
                zinb_loss_acc  += recon_loss.item()

                bank.enqueue(x0_k.detach().reshape(-1, n_genes))

            if not torch.isfinite(loss):
                print("[warn] NaN/Inf loss detected; skipping this batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        n_batches   = len(train_loader)
        train_loss /= n_batches
        current_lr  = scheduler.get_last_lr()[0]

        val_pcc, val_pcc10, val_pcc50, val_pcc200, \
            pred_mean, pred_std, true_mean, true_std, \
            val_mse, val_mae, svg_pcc20, svg_pcc50 = evaluate(
            model, val_loader, n_genes, device, svg_indices=svg_indices
        )

        if args.snapshot_epochs is not None and epoch in args.snapshot_epochs:
            snap_dir = Path(args.snapshot_dir or (out_dir / "snapshots"))
            snap_dir.mkdir(parents=True, exist_ok=True)
            *_, snap_pred_det, snap_true = evaluate(
                model, val_loader, n_genes, device,
                svg_indices=svg_indices, return_predictions=True,
            )
            snap_pred_stoch, _ = capture_stochastic(
                model, val_loader, device, n_genes, K=args.gen_per_spot,
            )
            d_now = (drift_loss_acc / max(1, len(train_loader))) if not is_warmup else float("nan")
            np.savez_compressed(
                snap_dir / f"snapshot_epoch{epoch:03d}.npz",
                pred=snap_pred_stoch.astype(np.float32),
                pred_det=snap_pred_det.astype(np.float32),
                true=snap_true.astype(np.float32),
                epoch=epoch,
                drift_loss=d_now,
                train_loss=train_loss,
            )
            model.train()
            print(f"[snapshot] saved epoch {epoch} -> {snap_dir}")

        phase = "warm " if is_warmup else "drift"
        if is_warmup:
            print(f"Epoch {epoch:03d} [{phase}] | Loss: {train_loss:.4f} | "
                  f"Val PCC: {val_pcc:.4f} | SVG-20: {svg_pcc20:.4f} | LR: {current_lr:.2e}")

            log_dict = {
                "epoch":           epoch,
                "phase":           0,
                "train/loss":      train_loss,
                "val/pcc_all":     val_pcc,
                "val/pcc_10":      val_pcc10,
                "val/pcc_50":      val_pcc50,
                "val/pcc_200":     val_pcc200,
                "val/svg_pcc_20":  svg_pcc20,
                "val/svg_pcc_50":  svg_pcc50,
                "train/lr":        current_lr,
                "diag/pred_mean":  pred_mean,
                "diag/pred_std":   pred_std,
                "diag/true_mean":  true_mean,
                "diag/true_std":   true_std,
                "val/mse_log2":    val_mse,
                "val/mae_log2":    val_mae,
            }
            if args.use_gate:
                log_dict["train/gate_loss"] = gate_loss_acc / n_batches
            if args.std_weight > 0:
                log_dict["train/std_loss"] = std_loss_acc / n_batches
            wandb.log(log_dict, step=epoch)

        else:
            d_avg = drift_loss_acc / n_batches
            z_avg = zinb_loss_acc  / n_batches
            g_avg = gate_loss_acc  / n_batches if args.use_gate else 0.0
            s_avg = std_loss_acc   / n_batches if args.std_weight > 0 else 0.0
            gate_str = f" gate={g_avg:.4f}" if args.use_gate else ""
            std_str  = f" std={s_avg:.4f}" if args.std_weight > 0 else ""
            print(f"Epoch {epoch:03d} [{phase}] | Loss: {train_loss:.4f} "
                  f"(drift={d_avg:.4f} zinb={z_avg:.4f}{std_str}{gate_str}) | "
                  f"Val PCC: {val_pcc:.4f} | SVG-20: {svg_pcc20:.4f} SVG-50: {svg_pcc50:.4f} | "
                  f"MSE={val_mse:.4f} MAE={val_mae:.4f} | LR: {current_lr:.2e}")

            log_dict = {
                "epoch":            epoch,
                "phase":            1,
                "train/loss":       train_loss,
                "train/drift_loss": d_avg,
                "train/zinb_loss":  z_avg,
                "val/pcc_all":      val_pcc,
                "val/pcc_10":       val_pcc10,
                "val/pcc_50":       val_pcc50,
                "val/pcc_200":      val_pcc200,
                "val/svg_pcc_20":   svg_pcc20,
                "val/svg_pcc_50":   svg_pcc50,
                "train/lr":         current_lr,
                "diag/pred_mean":   pred_mean,
                "diag/pred_std":    pred_std,
                "diag/true_mean":   true_mean,
                "diag/true_std":    true_std,
                "val/mse_log2":     val_mse,
                "val/mae_log2":     val_mae,
            }
            if args.use_gate:
                log_dict["train/gate_loss"] = gate_loss_acc / n_batches
            if args.std_weight > 0:
                log_dict["train/std_loss"] = std_loss_acc / n_batches
            wandb.log(log_dict, step=epoch)

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            torch.save({
                "state_dict": model.state_dict(),
                "val_pcc":    best_pcc,
                "fold":       args.fold,
                "test_slide": test_slide,
                "args":       vars(args),
            }, out_dir / "best_model.pt")

            wandb.run.summary["best_val_pcc"]   = best_pcc
            wandb.run.summary["best_epoch"]      = epoch

        def _ckpt():
            return {"state_dict": model.state_dict(), "val_pcc": val_pcc,
                    "train_loss": train_loss, "epoch": epoch, "fold": args.fold,
                    "test_slide": test_slide, "args": vars(args)}
        if not is_warmup and train_loss < best_loss:
            best_loss = train_loss
            torch.save(_ckpt(), out_dir / "min_loss_model.pt")
            wandb.run.summary["min_loss"]       = best_loss
            wandb.run.summary["min_loss_epoch"] = epoch
        torch.save(_ckpt(), out_dir / "last_model.pt")

    print(f"Fold {args.fold} finished. Best PCC: {best_pcc:.4f} | Min loss: {best_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
