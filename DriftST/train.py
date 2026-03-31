"""
DriftST 训练脚本
主要改动：
  1. 用 K-Means 对图像特征聚类，给每个 spot 分配 cluster_id
  2. ClusterAwareBank 按簇存取负样本
  3. patience 在 drift 启动时重置，避免 warmup→drift 切换时误触发早停
  4. bank 存 LN 后的值，取时不再重复做 LN
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

from dataset import build_datasets
from drift_step import ClusterAwareBank, multi_scale_drift_step, drift_loss_fn
from model import GenePredictor


# ─────────────────────────────────────────────────────────────
# 工具：给 dataset 加上 index 字段
# ─────────────────────────────────────────────────────────────

class IndexedDataset(Dataset):
    """
    包装原始 dataset，让每个 batch 里带上样本的全局下标（idx）
    这样才能查到对应的 cluster_id
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['idx'] = idx
        return item


# ─────────────────────────────────────────────────────────────
# 工具：K-Means 聚类
# ─────────────────────────────────────────────────────────────

def compute_clusters(dataset, n_clusters: int = 50) -> np.ndarray:
    """
    用图像特征（UNI+CONCH 的 2048 维）对所有 spot 做 K-Means 聚类
    返回: cluster_ids，shape (N,)，每个 spot 的簇编号
    """
    from sklearn.cluster import MiniBatchKMeans

    print(f"[聚类] 用图像特征做 K-Means，K={n_clusters}...")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    all_feats = []
    for batch in loader:
        all_feats.append(batch["z_img"].numpy())
    all_feats = np.concatenate(all_feats, axis=0)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=2048,
        n_init=3,
    )
    cluster_ids = kmeans.fit_predict(all_feats)
    print(f"[聚类] 完成：{n_clusters} 个簇，共 {len(cluster_ids)} 个 spot")
    return cluster_ids.astype(np.int64)


# ─────────────────────────────────────────────────────────────
# 基因聚类排列（原有逻辑，不变）
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

    corr_dist = pdist(exprs.T, metric="correlation")
    Z = linkage(corr_dist, method="ward")
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
        img      = batch["z_img"].to(device)
        nb_img   = batch["neighbor_zimg"].to(device)
        nb_valid = batch["neighbor_valid"].to(device)
        g_true   = batch["gene_expr"].to(device)

        x0, _ = model(img, nb_img, nb_valid)
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
    print(f"PCC-10={pcc_10:.4f} | PCC-50={pcc_50:.4f} | PCC-200={pcc_200:.4f} | PCC-all={pcc_all:.4f}")
    return pcc_all


# ─────────────────────────────────────────────────────────────
# Bank 预热
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def warmup_bank(model, loader, bank, cluster_ids_all, device, min_samples: int):
    """
    预热阶段：用真实基因值填充 bank
    ★ 对齐原版：bank 存真实数据，不存预测值
    """
    model.eval()
    collected = 0
    for batch in loader:
        g_true   = batch["gene_expr"].to(device)    # 真实基因值
        indices  = batch["idx"]

        # ★ 存 g_true 的 LN 值，不存模型预测
        g_true_ln = F.layer_norm(g_true, (g_true.shape[-1],))
        cids      = torch.tensor(cluster_ids_all[indices.numpy()], dtype=torch.long)
        bank.enqueue(g_true_ln.cpu(), cids)

        collected += g_true.shape[0]
        if collected >= min_samples:
            break

    print(f"Bank 预热完成：{collected} 个真实样本，"
          f"覆盖 {(bank.count > 0).sum().item()}/{bank.num_clusters} 个簇")
    model.train()


# ─────────────────────────────────────────────────────────────
# K 次 dropout 采样
# ─────────────────────────────────────────────────────────────

def sample_k_predictions(model, img, nb_img, nb_valid, K: int):
    preds = []
    for _ in range(K):
        x0, _ = model(img, nb_img, nb_valid)
        preds.append(x0.unsqueeze(1))
    return torch.cat(preds, dim=1)   # (B, K, n_genes)


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
    p.add_argument("--n_genes",          type=int,   default=200)
    p.add_argument("--input_dim",        type=int,   default=2048)
    p.add_argument("--hidden_dim",       type=int,   default=256)
    p.add_argument("--num_layers",       type=int,   default=4)
    p.add_argument("--num_heads",        type=int,   default=8)
    p.add_argument("--max_neighbors",    type=int,   default=6)
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
    p.add_argument("--bank_sample_size", type=int,   default=64)   # 每个 spot 取多少个正样本

    p.add_argument("--num_clusters",     type=int,   default=50)
    p.add_argument("--size_per_cluster", type=int,   default=256)

    args = p.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_datasets = build_datasets(data_dir=args.data_dir, fold=args.fold)
    raw_train_ds = raw_datasets[0]
    val_ds       = raw_datasets[1]

    train_ds = IndexedDataset(raw_train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers,
    )

    cluster_ids_all = compute_clusters(raw_train_ds, n_clusters=args.num_clusters)
    gene_order      = compute_gene_order(raw_train_ds, args.n_genes)

    model = GenePredictor(
        input_dim    = args.input_dim,
        hidden_dim   = args.hidden_dim,
        num_layers   = args.num_layers,
        num_heads    = args.num_heads,
        output_dim   = args.n_genes,
        dropout      = args.dropout,
        use_neighbor = True,
        max_neighbors= args.max_neighbors,
        gene_order   = gene_order,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量：{total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    bank = ClusterAwareBank(
        num_clusters     = args.num_clusters,
        size_per_cluster = args.size_per_cluster,
        feat_dim         = args.n_genes,
    )
    warmup_bank(model, train_loader, bank, cluster_ids_all, device,
                min_samples=args.bank_sample_size * 4)

    best_pcc   = -1.0
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss     = 0.0
        drift_loss_acc = 0.0
        mse_loss_acc   = 0.0
        is_warmup = (epoch < args.warm_epochs)

        # ★ drift 启动时重置 patience，给 drift 重新计时
        if epoch == args.warm_epochs:
            no_improve = 0
            best_pcc   = -1.0
            print("Drift 启动，重置早停计数器")

        for batch in train_loader:
            img      = batch["z_img"].to(device)
            nb_img   = batch["neighbor_zimg"].to(device)
            nb_valid = batch["neighbor_valid"].to(device)
            g_true   = batch["gene_expr"].to(device)
            indices  = batch["idx"]

            cids = torch.tensor(
                cluster_ids_all[indices.numpy()], dtype=torch.long
            )

            if is_warmup:
                x0, _ = model(img, nb_img, nb_valid)
                loss   = F.mse_loss(x0, g_true)

                # ★ warmup 阶段也更新 bank（存真实值）
                # 对齐原版：bank 每个 step 都更新，存真实数据
                g_true_ln = F.layer_norm(g_true.detach(), (g_true.shape[-1],))
                bank.enqueue(g_true_ln.cpu(), cids)

            else:
                K = args.gen_per_spot
                x0_k = sample_k_predictions(model, img, nb_img, nb_valid, K)

                # MSE loss
                g_true_k = g_true.unsqueeze(1).expand(-1, K, -1)
                mse_loss = F.mse_loss(x0_k, g_true_k)

                # drift 在 LN 空间做
                x0_k_ln = F.layer_norm(x0_k,  (x0_k.shape[-1],))       # (B, K, D)

                # ★ 正样本 = 同簇真实基因值（从 bank 取），shape (B, C_p, D)
                # 对齐原版：bank 存真实数据，sample 返回同类别真实样本作为正样本
                g_true_ln = F.layer_norm(g_true.detach(), (g_true.shape[-1],))
                pos_ln = bank.sample(cids, args.bank_sample_size, device) # (B, n, D)

                # ★ 负样本 = x0_k 自身（K个预测互相排斥）
                # 对齐原版：无需外部负样本，gen 自身互斥即可
                goal, goal_scaled, scale_inp = multi_scale_drift_step(
                    x      = x0_k_ln,
                    pos    = pos_ln,
                    R_list = tuple(args.R_list),
                    step   = args.drift_step,
                )

                d_loss = drift_loss_fn(x0_k_ln, goal_scaled, scale_inp)
                loss   = args.drift_weight * d_loss + \
                         (1.0 - args.drift_weight) * mse_loss

                drift_loss_acc += d_loss.item()
                mse_loss_acc   += mse_loss.item()

                # ★ 每个 step 都更新 bank（存真实值，对齐原版更新频率）
                bank.enqueue(g_true_ln.cpu(), cids)

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
        val_pcc = evaluate(model, val_loader, args.n_genes, device)

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
                "state_dict":  model.state_dict(),
                "val_pcc":     best_pcc,
                "fold":        args.fold,
                "gene_order":  gene_order,
                "cluster_ids": cluster_ids_all,
                "args":        vars(args),
            }, out_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"早停：{args.patience} epoch 无提升，最佳 PCC: {best_pcc:.4f}")
                break

    print(f"Fold {args.fold} 训练结束，最佳 PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()