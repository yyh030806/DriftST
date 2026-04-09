"""
DriftST — Drift Step

变更：
  - 删除 ClusterAwareBank，改为单纯的 PredictionBank（ring buffer，无聚类）
  - multi_scale_drift_step 新增 neg 参数（从 bank 取出的外部负样本）
  - 正样本 pos 形状为 (B, 1, D)（唯一匹配）
"""
import torch
import torch.nn.functional as F


def multi_scale_drift_step(
    x:      torch.Tensor,           # (B, K, D)      当前 dropout 预测，有梯度
    pos:    torch.Tensor,           # (B, 1, D)      detach，真实基因值（唯一匹配）
    neg:    torch.Tensor = None,    # (B, N_bank, D) detach，从 bank 取出的负样本
    R_list: tuple = (0.02, 0.05, 0.2),
    step:   float = 1.0,
) -> tuple:
    B, K, D = x.shape
    C_p     = pos.shape[1]          # 始终为 1
    old_x   = x.detach()           # (B, K, D)

    # ── 构建完整负样本集合 ────────────────────────────────────────────────────
    # 当前 K 个预测（互相排斥）+ 从 bank 取出的 N_bank 个
    if neg is not None:
        all_neg = torch.cat([old_x, neg], dim=1)  # (B, K+N_bank, D)
    else:
        all_neg = old_x                           # (B, K, D)
    N_neg = all_neg.shape[1]

    # targets = [all_neg | pos]
    targets   = torch.cat([all_neg, pos], dim=1)  # (B, N_neg+1, D)
    split_pos = N_neg                             # pos 从此索引开始

    w    = torch.ones(N_neg + C_p, device=x.device)
    w_2d = w.unsqueeze(0).expand(B, -1)          # (B, N_neg+C_p)

    with torch.no_grad():
        # dist: (B, K, N_neg+C_p)
        diff_all = targets.unsqueeze(1) - old_x.unsqueeze(2)  # (B, K, N_neg+C_p, D)
        dist     = diff_all.norm(dim=-1)                       # (B, K, N_neg+C_p)

        weighted_dist = dist * w_2d.unsqueeze(1)
        scale         = weighted_dist.mean() / w_2d.mean()
        scale_inp     = (scale / (D ** 0.5)).clamp(min=1e-3)

        old_x_sc    = old_x   / scale_inp
        targets_sc  = targets / scale_inp
        dist_normed = dist    / scale.clamp(min=1e-3)

        # self-mask：只对前 K 个当前预测生效（bank 样本不需要 self-mask）
        diag_mask = torch.zeros(K, N_neg + C_p, device=x.device)
        diag_mask[:, :K] = torch.eye(K, device=x.device) * 100.0
        dist_normed = dist_normed + diag_mask.unsqueeze(0)

        force_across_R = torch.zeros_like(old_x_sc)

        for R in R_list:
            logits   = -dist_normed / R
            aff_row  = F.softmax(logits, dim=-1)
            aff_col  = F.softmax(logits, dim=-2)
            affinity = (aff_row * aff_col).clamp(min=1e-6).sqrt()
            affinity = affinity * w_2d.unsqueeze(1)

            aff_neg = affinity[:, :, :split_pos]   # (B, K, N_neg)
            aff_pos = affinity[:, :, split_pos:]   # (B, K, 1)

            sum_pos     = aff_pos.sum(-1, keepdim=True)
            sum_neg     = aff_neg.sum(-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            r_coeff_pos =  aff_pos * sum_neg
            R_coeff     = torch.cat([r_coeff_neg, r_coeff_pos], dim=-1)

            total_force  = torch.einsum('bkn,bnd->bkd', R_coeff, targets_sc)
            total_coeffs = R_coeff.sum(-1, keepdim=True)
            total_force  = total_force - total_coeffs * old_x_sc

            f_norm         = (total_force ** 2).mean().clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force / f_norm

        goal_scaled = old_x_sc + step * force_across_R
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
# 存储 dropout 预测值，作为负样本 bank
# ─────────────────────────────────────────────────────────────

class PredictionBank:
    """
    单纯 ring buffer。
    存储 dropout 预测值（LN 后），对外提供负样本。

    enqueue：接收 (N, D) 张量，循环覆盖写入
    sample ：随机取 n 个，返回 (n, D)
    """

    def __init__(self, size: int = 8192, feat_dim: int = 300):
        self.size     = size
        self.feat_dim = feat_dim
        self.buf      = torch.zeros(size, feat_dim)
        self.ptr      = 0
        self.count    = 0

    @torch.no_grad()
    def enqueue(self, feat: torch.Tensor):
        """
        feat: (N, D)，调用前请确保已 detach
        包含 wrap-around 处理
        """
        feat = feat.detach().cpu().float()
        N    = feat.shape[0]
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
        """
        随机取 n 个样本
        返回：(n, D)，detach 张量
        """
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