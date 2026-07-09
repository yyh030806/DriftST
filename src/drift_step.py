"""One-step drifting utilities for DriftST.

This implementation uses a split-form drift step: positives and negatives are
normalized with separate softmax operations. This prevents the single matched
positive profile from being overwhelmed by many negative samples. Distances are
scaled by ``mean(distance) / sqrt(D)`` and the final velocity is averaged across
multiple drift radii.
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
    x:   (B, K, D) current dropout predictions
    pos: (B, N_pos, D) positive samples, usually one matched profile
    neg: (B, N_bank, D) optional external negative samples

    Returns:
      goal:        detached target in the original feature scale
      goal_scaled: detached target in the normalized scale used for loss
      scale_inp:   scalar normalization factor
    """
    B, K, D = x.shape
    old_x = x.detach()

    # Negative pool: current dropout predictions plus optional bank samples.
    if neg is not None:
        all_neg = torch.cat([old_x, neg], dim=1)   # (B, K + N_bank, D)
    else:
        all_neg = old_x                            # (B, K, D)
    N_neg = all_neg.shape[1]
    N_pos = pos.shape[1]

    with torch.no_grad():
        diff_neg = all_neg.unsqueeze(1) - old_x.unsqueeze(2)   # (B, K, N_neg, D)
        diff_pos = pos.unsqueeze(1)     - old_x.unsqueeze(2)   # (B, K, N_pos, D)

        dist_neg = diff_neg.norm(dim=-1)    # (B, K, N_neg)
        dist_pos = diff_pos.norm(dim=-1)    # (B, K, N_pos)

        # Normalize distances so that the normalized mean is approximately sqrt(D).
        all_dist_mean = (dist_neg.sum() + dist_pos.sum()) / (dist_neg.numel() + dist_pos.numel())
        scale = (all_dist_mean / math.sqrt(float(D))).clamp(min=1e-6)
        scale_inp = scale

        old_x_sc  = old_x  / scale_inp
        pos_sc    = pos    / scale_inp
        all_neg_sc = all_neg / scale_inp

        dist_neg_normed = dist_neg / scale_inp    # mean ≈ sqrt(D)
        dist_pos_normed = dist_pos / scale_inp

        # Mask the matching current prediction in the first K negative columns.
        diag_mask = torch.zeros(K, N_neg, device=x.device, dtype=dist_neg_normed.dtype)
        diag_mask[:, :K] = torch.eye(K, device=x.device, dtype=dist_neg_normed.dtype) * 1e6
        dist_neg_normed = dist_neg_normed + diag_mask.unsqueeze(0)

        v_agg = torch.zeros_like(old_x_sc)

        for R in R_list:
            temp_eff = R * math.sqrt(float(D))

            logits_pos = -dist_pos_normed / temp_eff        # (B, K, N_pos)
            A_pos = F.softmax(logits_pos, dim=-1)           # (B, K, N_pos)

            logits_neg = -dist_neg_normed / temp_eff        # (B, K, N_neg)
            A_neg = F.softmax(logits_neg, dim=-1)           # (B, K, N_neg)

            # V_R = drift_pos - drift_neg.
            drift_pos = torch.einsum('bkn,bnd->bkd', A_pos, pos_sc)
            drift_neg = torch.einsum('bkn,bnd->bkd', A_neg, all_neg_sc)

            v_R = drift_pos - drift_neg

            v_agg = v_agg + v_R

        force = v_agg / float(len(R_list))

        goal_scaled = old_x_sc + step * force
        goal        = goal_scaled * scale_inp

    return goal.detach(), goal_scaled.detach(), scale_inp.detach()


def drift_loss_fn(
    x:           torch.Tensor,
    goal_scaled: torch.Tensor,
    scale_inp:   torch.Tensor,
) -> torch.Tensor:
    x_scaled = x / scale_inp
    return F.mse_loss(x_scaled, goal_scaled)


class PredictionBank:
    """Ring buffer that stores stochastic predictions for negative sampling."""

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

        # If the incoming batch exceeds capacity, keep the most recent entries.
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


# Backward-compatible alias used by older checkpoints/scripts.
ClusterAwareBank = PredictionBank
