"""
model_mlp.py — DriftST GenePredictor（MLP 版）
替换 DiT，保持 forward 接口完全兼容：
  输入: img_emb, (neighbor_zimg, neighbor_valid 可选)
  输出: (x0, None)
"""
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# 邻居聚合（从原 model.py 保留，不改动）
# ─────────────────────────────────────────────────────────────

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.attn_w = nn.Linear(input_dim * 2, 1)
        self.proj   = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, center, neighbors, valid_mask):
        B, K, D = neighbors.shape
        q = center.unsqueeze(1).expand(-1, K, -1)
        w = self.attn_w(torch.cat([q, neighbors], dim=-1)).squeeze(-1)
        if valid_mask is not None:
            bool_mask = valid_mask.bool()
            has_valid = bool_mask.any(dim=-1, keepdim=True)
            safe_mask = bool_mask.masked_fill(~has_valid, True)
            w = w.masked_fill(~safe_mask, -1e9)
        w  = w.softmax(dim=-1).unsqueeze(-1)
        nb = (neighbors * w).sum(1)
        if valid_mask is not None:
            nb = nb * has_valid.float()
        return self.proj(nb)


# ─────────────────────────────────────────────────────────────
# MLP Block（带残差）
# ─────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    """单个残差 MLP block：Linear → LN → GELU → Dropout"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# ─────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────

class GenePredictor(nn.Module):
    def __init__(
        self,
        input_dim    : int   = 2048,
        hidden_dim   : int   = 256,
        num_layers   : int   = 4,
        num_heads    : int   = 8,      # 不用，保留参数兼容
        output_dim   : int   = 300,
        dropout      : float = 0.1,
        use_neighbor : bool  = True,
        max_neighbors: int   = 6,
        gene_order   = None,
    ):
        super().__init__()
        self.output_dim   = output_dim
        self.hidden_dim   = hidden_dim
        self.use_neighbor = use_neighbor

        # 基因重排序
        if gene_order is None:
            gene_order = list(range(output_dim))
        _order = torch.tensor(gene_order, dtype=torch.long)
        self.register_buffer("gene_order",     _order)
        self.register_buffer("gene_order_inv", torch.argsort(_order))

        # ── 图像特征投影 ──────────────────────────────────────
        self.img_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 邻居聚合（可选）──────────────────────────────────
        if use_neighbor:
            self.neighbor_agg = NeighborAggregator(input_dim, hidden_dim, dropout)
            fuse_in = hidden_dim * 2
        else:
            fuse_in = hidden_dim

        # ── 融合层（统一入口，不管有没有邻居）─────────────────
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 残差 MLP blocks ──────────────────────────────────
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── 输出头 ───────────────────────────────────────────
        self.head = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_emb, neighbor_zimg=None, neighbor_valid=None):
        """
        img_emb:        (B, input_dim)
        neighbor_zimg:  (B, K, input_dim) 或 None
        neighbor_valid: (B, K) 或 None

        返回:
          x0:     (B, output_dim)  基因表达预测
          x_feat: None             保持接口兼容
        """
        # 1. 图像特征投影
        c = self.img_proj(img_emb)                              # (B, hidden_dim)

        # 2. 邻居融合
        if self.use_neighbor and neighbor_zimg is not None:
            nb = self.neighbor_agg(img_emb, neighbor_zimg, neighbor_valid)
            c  = self.fuse(torch.cat([c, nb], dim=-1))          # (B, hidden_dim)
        else:
            c  = self.fuse(c)                                   # (B, hidden_dim)

        # 3. 残差 MLP blocks
        x = c
        for block in self.blocks:
            x = block(x)                                        # (B, hidden_dim)

        # 4. 输出
        x0 = self.head(x)                                       # (B, output_dim)
        x0 = x0[:, self.gene_order_inv]

        return x0, None


# 兼容旧版引用名
GeneMLP = GenePredictor