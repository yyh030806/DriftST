"""
model.py — DriftST GenePredictor

架构三层结构：
  1. Spatial Aggregation — 聚合邻居 spot 的视觉特征（可选）
  2. Gene-conditioned FiLM — 每个基因通过 learnable embedding 调制共享特征
  3. Gene Self-Attention — 基因间交互，建模 co-expression 关系

接口兼容：
  输入: img_emb, (neighbor_zimg, neighbor_valid 可选)
  输出: (x0, None)
"""
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Layer 1: 邻居空间聚合
# ─────────────────────────────────────────────────────────────

class NeighborAggregator(nn.Module):
    """Attention-weighted spatial neighbor aggregation (借鉴 STEM)"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.attn_w = nn.Linear(input_dim * 2, 1)
        self.proj = nn.Sequential(
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
        w = w.softmax(dim=-1).unsqueeze(-1)
        nb = (neighbors * w).sum(1)
        if valid_mask is not None:
            nb = nb * has_valid.float()
        return self.proj(nb)


# ─────────────────────────────────────────────────────────────
# Layer 2: Gene-conditioned FiLM Block
# ─────────────────────────────────────────────────────────────

class FiLMBlock(nn.Module):
    """
    AdaLN-Zero style FiLM conditioning.
    每个基因的 embedding 生成 (γ, β)，调制共享的 hidden features。
    初始化为恒等映射：γ=0, β=0 → (1+0)·h + 0 = h
    """
    def __init__(self, hidden_dim, d_emb, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film_gen = nn.Linear(d_emb, hidden_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: 初始化为恒等
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, h, gene_embs):
        """
        h:         (B, N_genes, hidden_dim)
        gene_embs: (N_genes, d_emb)
        """
        film_params = self.film_gen(gene_embs)          # (N_genes, hidden_dim*2)
        gamma, beta = film_params.chunk(2, dim=-1)      # 各 (N_genes, hidden_dim)

        h_norm = self.norm(h)
        # AdaLN-Zero: scale = 1 + γ, 初始时 γ=0 即不调制
        h_mod = (1.0 + gamma.unsqueeze(0)) * h_norm + beta.unsqueeze(0)

        return h + self.ffn(h_mod)                      # 残差连接


# ─────────────────────────────────────────────────────────────
# Layer 3: Gene Self-Attention Block
# ─────────────────────────────────────────────────────────────

class GeneSelfAttentionBlock(nn.Module):
    """
    标准 pre-norm Transformer block，用于基因间交互（借鉴 GenAR）。
    300 个 gene token 互相 attend，建模 co-expression 关系。
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, N_genes, hidden_dim)"""
        # Self-attention + residual
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        # FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────

class GenePredictor(nn.Module):
    """
    DriftST Gene Predictor

    信息流：
      Image Features (2048d)
            │
      ┌─────┴──────┐
      │   Neighbor Features (optional)
      └─── Spatial Aggregation ──┘      ← 借鉴 STEM
               │
          h (hidden_dim)
               │
       expand to (B, 300, hidden_dim)
               │
        Gene FiLM Blocks × N            ← FiLM / AdaLN-Zero
         (gene-specific modulation)
               │
        Gene Self-Attention              ← 借鉴 GenAR
         (co-expression modeling)
               │
        Linear → scalar per gene
               │
        Prediction (B, 300)
    """
    def __init__(
        self,
        input_dim:     int   = 2048,
        hidden_dim:    int   = 256,
        num_layers:    int   = 2,       # FiLM block 数量
        num_heads:     int   = 8,       # Gene Self-Attention heads
        output_dim:    int   = 300,
        dropout:       float = 0.1,
        use_neighbor:  bool  = False,
        max_neighbors: int   = 6,
        d_emb:         int   = 64,      # Gene embedding 维度
        gene_order     = None,
    ):
        super().__init__()
        self.output_dim   = output_dim
        self.hidden_dim   = hidden_dim
        self.use_neighbor = use_neighbor

        # ── 基因重排序 ────────────────────────────────────────
        if gene_order is None:
            gene_order = list(range(output_dim))
        _order = torch.tensor(gene_order, dtype=torch.long)
        self.register_buffer("gene_order",     _order)
        self.register_buffer("gene_order_inv", torch.argsort(_order))

        # ── Layer 0: 图像特征投影 ─────────────────────────────
        self.img_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Layer 1: 邻居空间聚合（可选）──────────────────────
        if use_neighbor:
            self.neighbor_agg = NeighborAggregator(input_dim, hidden_dim, dropout)
            fuse_in = hidden_dim * 2
        else:
            fuse_in = hidden_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Gene Embeddings ──────────────────────────────────
        self.gene_emb = nn.Embedding(output_dim, d_emb)

        # ── Layer 2: FiLM Blocks ─────────────────────────────
        self.film_blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, d_emb, dropout)
            for _ in range(num_layers)
        ])

        # ── Layer 3: Gene Self-Attention ─────────────────────
        self.gene_attn = GeneSelfAttentionBlock(hidden_dim, num_heads, dropout)

        # ── 输出头（共享，每个基因 hidden_dim → 1）───────────
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # FiLM generators 重新初始化为 0（AdaLN-Zero）
        for film in self.film_blocks:
            nn.init.zeros_(film.film_gen.weight)
            nn.init.zeros_(film.film_gen.bias)

    def forward(self, img_emb, neighbor_zimg=None, neighbor_valid=None):
        """
        img_emb:        (B, input_dim)
        neighbor_zimg:  (B, K, input_dim) 或 None
        neighbor_valid: (B, K) 或 None

        返回:
          x0:     (B, output_dim)  基因表达预测
          x_feat: None             保持接口兼容
        """
        B = img_emb.shape[0]

        # 1. 图像特征投影
        c = self.img_proj(img_emb)                              # (B, hidden_dim)

        # 2. 空间聚合
        if self.use_neighbor and neighbor_zimg is not None:
            nb = self.neighbor_agg(img_emb, neighbor_zimg, neighbor_valid)
            c = self.fuse(torch.cat([c, nb], dim=-1))           # (B, hidden_dim)
        else:
            c = self.fuse(c)                                    # (B, hidden_dim)

        # 3. 扩展到所有基因：共享起点
        h = c.unsqueeze(1).expand(-1, self.output_dim, -1)      # (B, 300, hidden_dim)
        h = h.contiguous()  # expand 后做 contiguous 避免后续问题

        # 4. Gene-conditioned FiLM：每个基因用自己的 embedding 调制特征
        gene_embs = self.gene_emb.weight                        # (300, d_emb)
        for film in self.film_blocks:
            h = film(h, gene_embs)                              # (B, 300, hidden_dim)

        # 5. Gene Self-Attention：基因间交互
        h = self.gene_attn(h)                                   # (B, 300, hidden_dim)

        # 6. 输出：每个基因独立映射到标量
        x0 = self.head(self.out_norm(h)).squeeze(-1)            # (B, 300)
        x0 = x0[:, self.gene_order_inv]

        return x0, None


# 兼容旧版引用名
GeneMLP = GenePredictor