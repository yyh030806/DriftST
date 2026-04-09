"""
DriftST 训练脚本

变更：
  1. 删除聚类（compute_clusters / IndexedDataset / cluster_ids）
  2. 负样本 = dropout 预测存入 ring buffer，每次取 256 个
  3. 正样本 = 唯一匹配的真实基因值（LN后），shape (B, 1, D)
  4. K=16 次 dropout 采样；首次保留梯度，其余 detach 节省显存
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from dataset import build_datasets
from drift_step import PredictionBank, multi_scale_drift_step, drift_loss_fn
from model import GenePredictor


# ─────────────────────────────────────────────────────────────
# 基因聚类排列
# ─────────────────────────────────────────────────────────────

def compute_gene_order(dataset, n_genes: int) -> list:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    print("预计算基因聚类排列...")
    exprs = []
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    for batch in loader:
        exprs.append(batch["gene_expr"].numpy())
    exprs = np.concatenate(exprs, axis=0)

    corr_dist  = pdist(exprs.T, metric="correlation")
    Z          = linkage(corr_dist, method="ward")
    gene_order = leaves_list(Z).tolist()
    print(f"基因聚类完成，共 {n_genes} 个基因")
    return gene_order


# ─────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, n_genes, device):
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        img    = batch["z_img"].to(device)
        g_true = batch["gene_expr"].to(device)

        x0, _ = model(img)
        all_pred.append(x0.cpu().numpy())
        all_true.append(g_true.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    print(f"pred mean={all_pred.mean():.4f}, std={all_pred.std():.4f}")
    print(f"true mean={all_true.mean():.4f}, std={all_true.std():.4f}")

    pccs = []
    for i in range(n_genes):
        pcc, _ = pearsonr(all_pred[:, i], all_true[:, i])
        if not np.isnan(pcc):
            pccs.append(pcc)

    if not pccs:
        return 0.0

    pccs_sorted = sorted(pccs, reverse=True)
    pcc_10  = np.mean(pccs_sorted[:10])
    pcc_50  = np.mean(pccs_sorted[:min(50,  len(pccs_sorted))])
    pcc_200 = np.mean(pccs_sorted[:min(200, len(pccs_sorted))])
    pcc_all = np.mean(pccs)
    print(f"PCC-10={pcc_10:.4f} | PCC-50={pcc_50:.4f} | "
          f"PCC-200={pcc_200:.4f} | PCC-all={pcc_all:.4f}")
    return pcc_all


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
        img = batch["z_img"].to(device)

        preds = []
        for _ in range(K):
            x0, _ = model(img)
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

def sample_k_predictions(model, img, K: int):
    """
    只有第一次前向保留梯度，其余 K-1 次 detach。
    drift_loss_fn 的 MSE 虽然对整个 x0_k 计算，
    但实际梯度只流经第一个预测 → 显存节约 K 倍。
    """
    preds = []

    x0, _ = model(img)                      # 保留梯度
    preds.append(x0.unsqueeze(1))

    with torch.no_grad():
        for _ in range(K - 1):
            x0, _ = model(img)              # detach，仅作负样本参考点
            preds.append(x0.unsqueeze(1))

    return torch.cat(preds, dim=1)          # (B, K, n_genes)


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

    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--wd",               type=float, default=1e-4)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--warm_epochs",      type=int,   default=10)
    p.add_argument("--drift_weight",     type=float, default=0.15)
    p.add_argument("--gen_per_spot",     type=int,   default=16)

    p.add_argument("--R_list",           type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--drift_step",       type=float, default=1.0)
    p.add_argument("--bank_size",        type=int,   default=8192)
    p.add_argument("--bank_sample_size", type=int,   default=256)

    args = p.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    train_loader = DataLoader(
        raw_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers,
    )

    gene_order = compute_gene_order(raw_train_ds, n_genes)

    model = GenePredictor(
        input_dim  = args.input_dim,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        num_heads  = args.num_heads,
        output_dim = n_genes,
        dropout    = args.dropout,
        gene_order = gene_order,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量：{total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
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

    best_pcc   = -1.0
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss     = 0.0
        drift_loss_acc = 0.0
        mse_loss_acc   = 0.0
        is_warmup = (epoch < args.warm_epochs)

        if epoch == args.warm_epochs:
            no_improve = 0
            best_pcc   = -1.0
            print("Drift 启动，重置早停计数器")

        for batch in train_loader:
            img    = batch["z_img"].to(device)
            g_true = batch["gene_expr"].to(device)
            B      = img.shape[0]

            if is_warmup:
                x0, _ = model(img)
                loss   = F.mse_loss(x0, g_true)

                # warmup 阶段也填充 bank
                with torch.no_grad():
                    preds = [model(img)[0] for _ in range(args.gen_per_spot)]
                    x0_k_w = torch.stack(preds, dim=1)
                    bank.enqueue(x0_k_w.reshape(-1, n_genes))

            else:
                K    = args.gen_per_spot
                x0_k = sample_k_predictions(model, img, K)      # (B, K, n_genes)

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
                loss   = d_loss

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
        val_pcc = evaluate(model, val_loader, n_genes, device)

        phase = "warm " if is_warmup else "drift"
        if is_warmup:
            print(f"Epoch {epoch:03d} [{phase}] | Loss: {train_loss:.4f} | "
                  f"Val PCC: {val_pcc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        else:
            d_avg = drift_loss_acc / n_batches
            m_avg = mse_loss_acc   / n_batches
            print(f"Epoch {epoch:03d} [{phase}] | Loss: {train_loss:.4f} "
                  f"(drift={d_avg:.4f} mse={m_avg:.4f}) | "
                  f"Val PCC: {val_pcc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_pcc > best_pcc:
            best_pcc   = val_pcc
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "val_pcc":    best_pcc,
                "fold":       args.fold,
                "test_slide": test_slide,
                "gene_order": gene_order,
                "args":       vars(args),
            }, out_dir / "best_model.pt")
            
    print(f"Fold {args.fold} 训练结束，最佳 PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()