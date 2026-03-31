"""
DriftST — Drift Step（对齐原版 Drifting Model）

对齐说明：
  正样本(pos): 同簇的真实基因值（从 bank 取，per-batch-element）
  负样本(neg): K 个 dropout 生成样本本身（互相排斥，无需外部 neg）
  scale_inp  : 去掉原来自己加的 max=10.0，与原版一致

gen  shape: (B, K, D)
pos  shape: (B, C_p, D)   ← 每个 spot 从同簇 bank 取 C_p 个真实样本
"""
import torch
import torch.nn.functional as F


def multi_scale_drift_step(
    x:      torch.Tensor,          # (B, K, D)   有梯度，K 个 dropout 采样
    pos:    torch.Tensor,          # (B, C_p, D) detach，同簇真实基因值
    R_list: tuple = (0.02, 0.05, 0.2),
    step:   float = 1.0,
) -> tuple:
    """
    正样本 = pos（同簇真实值，从 bank 取）
    负样本 = x 自身的 K 个预测互相排斥（无需外部 neg）

    Returns:
        goal        : (B, K, D)  原始空间目标（debug 用）
        goal_scaled : (B, K, D)  scaled 空间目标（★ 用于计算 loss）
        scale_inp   : scalar     缩放因子（detach）
    """
    B, K, D  = x.shape
    C_p      = pos.shape[1]
    old_x    = x.detach()                                       # (B, K, D)

    # targets: [gen(K) | pos(C_p)]，shape (B, K+C_p, D)
    # gen 部分互相排斥，pos 部分吸引当前预测
    targets = torch.cat([
        old_x,                                                   # (B, K,   D)
        pos,                                                     # (B, C_p, D)
    ], dim=1)                                                    # (B, K+C_p, D)

    split_pos = K      # pos 从此索引开始

    # weights：gen 和 pos 权重都是 1（与原版一致）
    w    = torch.ones(K + C_p, device=x.device)                 # (K+C_p,)
    w_2d = w.unsqueeze(0).expand(B, -1)                         # (B, K+C_p)

    with torch.no_grad():
        # dist: (B, K, K+C_p)
        diff_all    = targets.unsqueeze(1) - old_x.unsqueeze(2) # (B, K, K+C_p, D)
        dist        = diff_all.norm(dim=-1)                     # (B, K, K+C_p)

        # scale：全局标量，与原版完全一致
        weighted_dist = dist * w_2d.unsqueeze(1)                # (B, K, K+C_p)
        scale         = weighted_dist.mean() / w_2d.mean()
        # ★ 去掉之前自己加的 max=10.0，与原版对齐（原版只有 min=1e-3）
        scale_inp     = (scale / (D ** 0.5)).clamp(min=1e-3)

        old_x_sc    = old_x   / scale_inp                      # (B, K, D)
        targets_sc  = targets / scale_inp                       # (B, K+C_p, D)
        dist_normed = dist    / scale.clamp(min=1e-3)           # (B, K, K+C_p)

        # self-mask：gen[i] 不排斥自身（只排斥其他 gen[j]，j≠i）
        diag_mask = torch.zeros(K, K + C_p, device=x.device)
        diag_mask[:, :K] = torch.eye(K, device=x.device) * 100.0
        dist_normed = dist_normed + diag_mask.unsqueeze(0)      # (B, K, K+C_p)

        force_across_R = torch.zeros_like(old_x_sc)            # (B, K, D)

        for R in R_list:
            logits = -dist_normed / R                           # (B, K, K+C_p)

            # ★ 双向 softmax 几何平均（原版核心设计）
            aff_row  = F.softmax(logits, dim=-1)                # 对 target 维度
            aff_col  = F.softmax(logits, dim=-2)                # 对 query 维度
            affinity = (aff_row * aff_col).clamp(min=1e-6).sqrt()
            affinity = affinity * w_2d.unsqueeze(1)             # (B, K, K+C_p)

            aff_neg  = affinity[:, :, :split_pos]               # (B, K, K)   gen互斥
            aff_pos  = affinity[:, :, split_pos:]               # (B, K, C_p) 真实值吸引

            sum_pos      = aff_pos.sum(-1, keepdim=True)        # (B, K, 1)
            sum_neg      = aff_neg.sum(-1, keepdim=True)        # (B, K, 1)
            r_coeff_neg  = -aff_neg * sum_pos                   # 排斥
            r_coeff_pos  =  aff_pos * sum_neg                   # 吸引
            R_coeff      = torch.cat([r_coeff_neg, r_coeff_pos], dim=-1)  # (B, K, K+C_p)

            total_force  = torch.einsum('bkn,bnd->bkd', R_coeff, targets_sc)
            total_coeffs = R_coeff.sum(-1, keepdim=True)        # (B, K, 1)
            total_force  = total_force - total_coeffs * old_x_sc

            # ★ 全局均值归一化（原版做法）
            f_norm         = (total_force ** 2).mean().clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force / f_norm

        goal_scaled = old_x_sc + step * force_across_R         # (B, K, D)
        goal        = goal_scaled * scale_inp                   # (B, K, D)

    return goal.detach(), goal_scaled.detach(), scale_inp.detach()


def drift_loss_fn(
    x:           torch.Tensor,    # (B, K, D) 有梯度
    goal_scaled: torch.Tensor,    # (B, K, D) detach
    scale_inp:   torch.Tensor,    # scalar    detach
) -> torch.Tensor:
    """在 scaled 空间计算 MSE，与原版完全对齐"""
    x_scaled = x / scale_inp
    return F.mse_loss(x_scaled, goal_scaled)


# ─────────────────────────────────────────────────────────────
# ClusterAwareBank：按图像相似性簇分组，存真实基因值
# ─────────────────────────────────────────────────────────────

class ClusterAwareBank:
    """
    存储真实基因值（g_true），按图像相似性簇分桶。

    对齐原版设计：
      - 原版 bank 存真实数据（real data），按类别分桶
      - sample 返回同类别的真实样本作为正样本
      - 每个训练 step 都更新 bank（push 最新真实样本）

    sample 返回 (B, n, D)：
      每个 batch element 从自己所在簇取 n 个真实样本
      → 不同 spot 取到不同的正样本，更精准
    """

    def __init__(self, num_clusters: int, size_per_cluster: int = 256, feat_dim: int = 200):
        self.num_clusters     = num_clusters
        self.size_per_cluster = size_per_cluster
        self.feat_dim         = feat_dim

        self.buf   = torch.zeros(num_clusters, size_per_cluster, feat_dim)
        self.ptr   = torch.zeros(num_clusters, dtype=torch.long)
        self.count = torch.zeros(num_clusters, dtype=torch.long)

    @torch.no_grad()
    def enqueue(self, feat: torch.Tensor, cluster_ids: torch.Tensor):
        """
        每个训练 step 都调用，存入真实基因值

        feat:        (B, D)  LN 空间的真实基因值 g_true_ln
        cluster_ids: (B,)    每个 spot 的簇 id
        """
        feat        = feat.detach().cpu().float()
        cluster_ids = cluster_ids.cpu().long()

        for i in range(feat.shape[0]):
            cid              = int(cluster_ids[i])
            idx              = int(self.ptr[cid])
            self.buf[cid, idx] = feat[i]
            self.ptr[cid]    = (idx + 1) % self.size_per_cluster
            self.count[cid]  = min(int(self.count[cid]) + 1, self.size_per_cluster)

    def sample(self, cluster_ids: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        """
        返回 (B, n, D)：每个 spot 从自己所在簇取 n 个真实样本

        cluster_ids: (B,)
        返回: (B, n, D)
        """
        cluster_ids = cluster_ids.cpu().long()
        B           = cluster_ids.shape[0]
        results     = []

        for i in range(B):
            cid   = int(cluster_ids[i])
            valid = int(self.count[cid])
            if valid == 0:
                # 该簇还没有样本，用零向量占位
                results.append(torch.zeros(n, self.feat_dim))
            else:
                idx = torch.randint(0, valid, (n,))
                results.append(self.buf[cid, idx])              # (n, D)

        return torch.stack(results, dim=0).to(device)           # (B, n, D)

    @property
    def total_count(self) -> int:
        return int(self.count.sum())


PredictionBank = ClusterAwareBank