"""
DriftST — Drift Step（split form 版）

设计：
  1. pos / neg 分开 softmax（split form，Kaiming 论文附录 ablation 变体）
     - 避免 N_pos=1 被 N_neg=264 淹没
     - N_pos=1 时 A_pos ≡ 1，drift_pos 直接等于 pos（真值）
  2. 温度归一化对齐论文：scale = mean(dist) / sqrt(D)
  3. drift 公式：V = drift_pos - drift_neg
  4. 多尺度平均（不在循环内 unit-norm，让 loss 有信息量）
  5. PredictionBank 不变
"""
import torch
import torch.nn.functional as F
import math


def multi_scale_drift_step(
    x, pos, neg=None,
    R_list=(0.02, 0.05, 0.2),
    step=1.0,
):
    """
    x:   (B, K, D)  当前 dropout 预测
    pos: (B, N_pos, D)  正样本（通常 N_pos=1）
    neg: (B, N_bank, D) 或 None  外部负样本

    返回:
      goal:        (B, K, D) detach
      goal_scaled: (B, K, D) detach（loss 用）
      scale_inp:   scalar detach
    """
    B, K, D = x.shape
    old_x = x.detach()

    # 负样本池：[batch内K个dropout预测, bank外部负样本]
    if neg is not None:
        all_neg = torch.cat([old_x, neg], dim=1)   # (B, K + N_bank, D)
    else:
        all_neg = old_x                            # (B, K, D)
    N_neg = all_neg.shape[1]
    N_pos = pos.shape[1]

    with torch.no_grad():
        # ─── 距离 ────────────────────────────────────────────
        # 对 neg 和 pos 分别算距离
        diff_neg = all_neg.unsqueeze(1) - old_x.unsqueeze(2)   # (B, K, N_neg, D)
        diff_pos = pos.unsqueeze(1)     - old_x.unsqueeze(2)   # (B, K, N_pos, D)

        dist_neg = diff_neg.norm(dim=-1)    # (B, K, N_neg)
        dist_pos = diff_pos.norm(dim=-1)    # (B, K, N_pos)

        # ─── 特征归一化 S（对齐论文 Eq. 20-21）────────────────
        # scale 用所有距离的均值除以 sqrt(D)，让 dist_normed mean ≈ sqrt(D)
        all_dist_mean = (dist_neg.sum() + dist_pos.sum()) / (dist_neg.numel() + dist_pos.numel())
        scale = (all_dist_mean / math.sqrt(float(D))).clamp(min=1e-6)
        scale_inp = scale

        old_x_sc  = old_x  / scale_inp
        pos_sc    = pos    / scale_inp
        all_neg_sc = all_neg / scale_inp

        dist_neg_normed = dist_neg / scale_inp    # mean ≈ sqrt(D)
        dist_pos_normed = dist_pos / scale_inp

        # 自身对角 mask：前 K 列是 batch 内 K 个 dropout 预测，query i 不要把自己当 neg
        diag_mask = torch.zeros(K, N_neg, device=x.device, dtype=dist_neg_normed.dtype)
        diag_mask[:, :K] = torch.eye(K, device=x.device, dtype=dist_neg_normed.dtype) * 1e6
        dist_neg_normed = dist_neg_normed + diag_mask.unsqueeze(0)

        # ─── 多尺度 drift 累积 ─────────────────────────────────
        v_agg = torch.zeros_like(old_x_sc)

        for R in R_list:
            # 论文 Eq. 22: ρ̃ = ρ · sqrt(D)
            temp_eff = R * math.sqrt(float(D))

            # pos 内部 softmax（N_pos=1 时恒为 1）
            logits_pos = -dist_pos_normed / temp_eff        # (B, K, N_pos)
            A_pos = F.softmax(logits_pos, dim=-1)           # (B, K, N_pos)

            # neg 内部 softmax
            logits_neg = -dist_neg_normed / temp_eff        # (B, K, N_neg)
            A_neg = F.softmax(logits_neg, dim=-1)           # (B, K, N_neg)

            # V_R = drift_pos - drift_neg
            # drift_pos: query 想靠近的方向（N_pos=1 时就是 pos 本身）
            # drift_neg: query 最像的负样本方向（要推开的）
            drift_pos = torch.einsum('bkn,bnd->bkd', A_pos, pos_sc)
            drift_neg = torch.einsum('bkn,bnd->bkd', A_neg, all_neg_sc)

            v_R = drift_pos - drift_neg

            v_agg = v_agg + v_R

        # 多尺度平均
        force = v_agg / float(len(R_list))

        goal_scaled = old_x_sc + step * force
        goal        = goal_scaled * scale_inp

    return goal.detach(), goal_scaled.detach(), scale_inp.detach()


def drift_loss_fn(
    x:           torch.Tensor,   # (B, K, D) 有梯度
    goal_scaled: torch.Tensor,   # (B, K, D) detach
    scale_inp:   torch.Tensor,   # scalar    detach
) -> torch.Tensor:
    x_scaled = x / scale_inp
    return F.mse_loss(x_scaled, goal_scaled)


# ─────────────────────────────────────────────────────────────
# PredictionBank：单纯 ring buffer，无聚类
# ─────────────────────────────────────────────────────────────

class PredictionBank:
    """
    单纯 ring buffer。
    存储 dropout 预测值（LN 后），对外提供负样本。
    """

    def __init__(self, size: int = 8192, feat_dim: int = 300):
        self.size     = size
        self.feat_dim = feat_dim
        self.buf      = torch.zeros(size, feat_dim)
        self.ptr      = 0
        self.count    = 0

    @torch.no_grad()
    def enqueue(self, feat: torch.Tensor):
        feat = feat.detach().cpu().float()
        N    = feat.shape[0]

        # 单次入队数量 >= 缓冲容量：只保留最近 size 个，重置指针
        if N >= self.size:
            self.buf[:] = feat[-self.size:]
            self.ptr    = 0
            self.count  = self.size
            return

        end  = self.ptr + N

        if end <= self.size:
            self.buf[self.ptr:end] = feat
        else:
            first = self.size - self.ptr
            self.buf[self.ptr:]          = feat[:first]
            self.buf[:end - self.size]   = feat[first:]

        self.ptr   = end % self.size
        self.count = min(self.count + N, self.size)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        valid = min(self.count, self.size)
        if valid == 0:
            return torch.zeros(n, self.feat_dim, device=device)
        idx = torch.randint(0, valid, (n,))
        return self.buf[idx].to(device)

    @property
    def total_count(self) -> int:
        return self.count


# 旧版兼容
ClusterAwareBank = PredictionBank