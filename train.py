"""
DriftST 训练脚本

变更：
  1. 删除聚类（compute_clusters / IndexedDataset / cluster_ids）
  2. 负样本 = dropout 预测存入 ring buffer，每次取 256 个
  3. 正样本 = 唯一匹配的真实基因值（LN后），shape (B, 1, D)
  4. K=16 次 dropout 采样；首次保留梯度，其余 detach 节省显存
  5. [新增] wandb 记录 loss / PCC 曲线
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import build_datasets
from src.drift_step import PredictionBank, multi_scale_drift_step, drift_loss_fn
from src.model import GenePredictor
from src.evaluation import evaluate




# ─────────────────────────────────────────────────────────────
# Bank 预热
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def warmup_bank(model, loader, bank, device, K: int, min_samples: int):
    """
    用 K 次 dropout 前向填充 bank
    保持 train 模式（dropout 需要激活以产生多样性）
    """
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

    print(f"Bank 预热完成：{collected} 个样本 "
          f"(bank.count={bank.total_count}/{bank.size})")


# ─────────────────────────────────────────────────────────────
# K 次 dropout 采样（显存节约版）
# ─────────────────────────────────────────────────────────────

def sample_k_predictions(model, img, K: int, nb_img=None, nb_valid=None):
    """
    只有第一次前向保留梯度，其余 K-1 次 detach。
    返回:
      x0_k:      (B, K, n_genes)
      gate_info: dict or None  来自第一次前向（有梯度的那次）
    """
    preds = []

    x0, gate_info = model(img, nb_img, nb_valid)
    preds.append(x0.unsqueeze(1))

    with torch.no_grad():
        for _ in range(K - 1):
            x0, _ = model(img, nb_img, nb_valid)
            preds.append(x0.unsqueeze(1))

    return torch.cat(preds, dim=1), gate_info


# ─────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────

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
                   help="Bio-guided Attention 层数")
    p.add_argument("--use_gate",        action="store_true",
                   help="启用 Progressive Gene Gating")
    p.add_argument("--gate_weight",     type=float, default=0.1,
                   help="gate sparsity loss 的权重")
    p.add_argument("--gate_targets",    type=float, nargs="+", default=None,
                   help="每层目标保留率，如 0.8 0.5")
    p.add_argument("--gate_entropy_weight", type=float, default=0.1,
                   help="gate 熵正则权重，推动二值化")
    p.add_argument("--balance_slides",  action="store_true",
                   help="按切片做均衡采样：每个切片在 epoch 内期望命中次数相等")

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
                   help="启用邻居空间聚合")
    p.add_argument("--max_neighbors",   type=int,   default=6)

    # ── wandb 相关参数 ──────────────────────────────────────
    p.add_argument("--wandb_project",    type=str,   default="DriftST")
    p.add_argument("--wandb_name",       type=str,   default=None,
                   help="run 名称，默认为 fold-{fold}")
    p.add_argument("--wandb_offline",    action="store_true",
                   help="离线模式（服务器无网时使用）")

    args = p.parse_args()

    # ── 初始化 wandb ────────────────────────────────────────
    if args.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    run_name = args.wandb_name or f"fold-{args.fold}"
    wandb.init(
        project = args.wandb_project,
        name    = run_name,
        config  = vars(args),          # 自动记录所有超参数
    )

    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 SVG 排序（若存在）
    svg_indices = None
    svg_path = Path(args.data_dir) / "svg_ranking.json"
    if svg_path.exists():
        with open(svg_path) as f:
            svg_data = json.load(f)
        with open(Path(args.data_dir) / "gene_names.json") as f:
            gene_names = json.load(f)
        gene2idx = {g: i for i, g in enumerate(gene_names)}
        svg_indices = [gene2idx[g] for g in svg_data["ranking"] if g in gene2idx]
        print(f"SVG ranking 加载完成，共 {len(svg_indices)} 个基因 (来自 {svg_path.name})")
    else:
        print("未找到 svg_ranking.json，跳过 SVG PCC 评估")

    raw_train_ds, val_ds, meta = build_datasets(
        data_dir=args.data_dir, fold=args.fold
    )

    data_n_genes = meta["n_genes"]
    if args.n_genes != data_n_genes:
        raise ValueError(
            f"[n_genes 不匹配] --n_genes={args.n_genes} 但数据实际有 "
            f"{data_n_genes} 个基因。请将 --n_genes 改为 {data_n_genes}。"
        )
    n_genes    = data_n_genes
    test_slide = meta["test_slide"]

    print(f"数据信息: n_genes={n_genes}, test_slide={test_slide}, "
          f"n_train={meta['n_train']}, n_test={meta['n_test']}")

    if args.balance_slides:
        from collections import Counter
        slide_counts = Counter(raw_train_ds.slide_ids)
        sample_weights = [1.0 / slide_counts[s] for s in raw_train_ds.slide_ids]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(raw_train_ds), replacement=True,
        )
        print(f"启用切片均衡采样: {len(slide_counts)} 个切片, "
              f"spot 数 min={min(slide_counts.values())}, max={max(slide_counts.values())}")
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

    # ── 计算基因调控关系矩阵（Pearson 相关）并加载 ─────────
    print("计算基因共表达矩阵（Pearson 相关）...")
    # 直接读常驻内存的 gene_expr，避开邻居 mmap 读
    train_expr = (raw_train_ds.gene_expr.numpy()
                  if hasattr(raw_train_ds.gene_expr, "numpy")
                  else np.asarray(raw_train_ds.gene_expr))   # (N_train, n_genes)
    R = np.corrcoef(train_expr.T)                           # (n_genes, n_genes)
    R = np.nan_to_num(R, nan=0.0)                           # 处理常量基因
    model.load_bio_bias(R)
    print(f"共表达矩阵加载完成，shape={R.shape}, "
          f"mean={R.mean():.4f}, nonzero={np.count_nonzero(np.abs(R) > 0.3)}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量：{total_params / 1e6:.2f}M")
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

    for epoch in range(args.epochs):
        model.train()
        train_loss        = 0.0
        drift_loss_acc    = 0.0
        mse_loss_acc      = 0.0
        gate_loss_acc     = 0.0
        is_warmup = (epoch < args.warm_epochs)

        if epoch == args.warm_epochs:
            best_pcc = -1.0
            print("Drift 启动")

        for batch in train_loader:
            img      = batch["z_img"].to(device)
            g_true   = batch["gene_expr"].to(device)
            nb_img   = batch["neighbor_zimg"].to(device)
            nb_valid = batch["neighbor_valid"].to(device)
            B        = img.shape[0]

            if is_warmup:
                x0, gate_info = model(img, nb_img, nb_valid)
                loss = F.mse_loss(x0, g_true)

                # gate sparsity loss
                if gate_info is not None:
                    g_loss = gate_info.get('gate_sparsity_loss')
                    if g_loss is not None:
                        loss = loss + args.gate_weight * g_loss
                        gate_loss_acc += g_loss.item()

                # warmup 阶段也填充 bank
                with torch.no_grad():
                    preds = [model(img, nb_img, nb_valid)[0] for _ in range(args.gen_per_spot)]
                    x0_k_w = torch.stack(preds, dim=1)
                    bank.enqueue(x0_k_w.reshape(-1, n_genes))

            else:
                K    = args.gen_per_spot
                x0_k, gate_info = sample_k_predictions(model, img, K, nb_img, nb_valid)  # (B, K, n_genes)

                # MSE loss：基于第一个预测
                mse_loss = F.mse_loss(x0_k[:, 0, :], g_true)

                # 正样本：唯一匹配，原始空间 (B, 1, D)
                pos = g_true.detach().unsqueeze(1)

                # 负样本：从 bank 取 256 个原始预测，所有 batch element 共享
                neg_bank = bank.sample(args.bank_sample_size, device)         # (256, D)
                neg_bank = neg_bank.unsqueeze(0).expand(B, -1, -1)            # (B, 256, D)

                goal, goal_scaled, scale_inp = multi_scale_drift_step(
                    x      = x0_k,
                    pos    = pos,
                    neg    = neg_bank,
                    R_list = tuple(args.R_list),
                    step   = args.drift_step,
                )

                d_loss = drift_loss_fn(x0_k, goal_scaled, scale_inp)
                loss   = args.drift_weight * d_loss + (1 - args.drift_weight) * mse_loss

                # gate sparsity loss
                if gate_info is not None:
                    g_loss = gate_info.get('gate_sparsity_loss')
                    if g_loss is not None:
                        loss = loss + args.gate_weight * g_loss
                        gate_loss_acc += g_loss.item()

                drift_loss_acc += d_loss.item()
                mse_loss_acc   += mse_loss.item()

                # bank 更新：本批次 B*K 个原始预测全部 enqueue
                bank.enqueue(x0_k.detach().reshape(-1, n_genes))

            if not torch.isfinite(loss):
                print("[防御] NaN/Inf，跳过 batch")
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

        # evaluate 返回 12 个指标
        val_pcc, val_pcc10, val_pcc50, val_pcc200, \
            pred_mean, pred_std, true_mean, true_std, \
            val_mse, val_mae, svg_pcc20, svg_pcc50 = evaluate(
            model, val_loader, n_genes, device, svg_indices=svg_indices
        )

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
            wandb.log(log_dict, step=epoch)

        else:
            d_avg = drift_loss_acc / n_batches
            m_avg = mse_loss_acc   / n_batches
            g_avg = gate_loss_acc  / n_batches if args.use_gate else 0.0
            gate_str = f" gate={g_avg:.4f}" if args.use_gate else ""
            print(f"Epoch {epoch:03d} [{phase}] | Loss: {train_loss:.4f} "
                  f"(drift={d_avg:.4f} mse={m_avg:.4f}{gate_str}) | "
                  f"Val PCC: {val_pcc:.4f} | SVG-20: {svg_pcc20:.4f} SVG-50: {svg_pcc50:.4f} | "
                  f"MSE={val_mse:.4f} MAE={val_mae:.4f} | LR: {current_lr:.2e}")

            log_dict = {
                "epoch":            epoch,
                "phase":            1,
                "train/loss":       train_loss,
                "train/drift_loss": d_avg,
                "train/mse_loss":   m_avg,
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

            # 在 wandb 里标记最佳 checkpoint
            wandb.run.summary["best_val_pcc"]   = best_pcc
            wandb.run.summary["best_epoch"]      = epoch

    print(f"Fold {args.fold} 训练结束，最佳 PCC: {best_pcc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()