"""
model.py — DriftST GenePredictor（DiT Transformer 版）
仿照 STEM 架构：每个基因是一个 token，图像特征通过 AdaLN-Zero 注入每一层
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    """
    AdaLN 调制：让图像条件动态控制每一层的缩放和偏移
    x:     (B, num_genes, hidden_dim)
    shift: (B, hidden_dim)
    scale: (B, hidden_dim)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────────────────────────
# 基因身份嵌入
# ─────────────────────────────────────────────────────────────

class GeneIdentityEmbedding(nn.Module):
    """
    给每个基因一个可学习的身份向量（就像词向量）
    让模型知道"这个 token 是第 i 号基因"
    """
    def __init__(self, num_genes: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(num_genes, hidden_dim), requires_grad=True
        )
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, B: int) -> torch.Tensor:
        # 每个 batch 里的 spot 都用同一套基因身份向量
        return self.embedding.unsqueeze(0).expand(B, -1, -1)  # (B, num_genes, hidden_dim)


# ─────────────────────────────────────────────────────────────
# DiT Block（核心模块）
# ─────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    一个 DiT block，包含：
      1. 自注意力（基因之间互相交流）
      2. FFN（每个基因自己深度处理）
      3. AdaLN-Zero（图像特征控制每一步的缩放/偏移/门控）

    adaLN-Zero 的特点：所有调制参数零初始化
    → 训练开始时整个 block 是恒等映射，梯度稳定
    """
    def __init__(self, hidden_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # adaLN-Zero：由图像条件生成 6 个调制参数
        # shift_a, scale_a, gate_a（注意力分支）
        # shift_m, scale_m, gate_m（FFN 分支）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        # ★ 零初始化：训练开始时调制量全为0，block 输出等于输入
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_genes, hidden_dim)  基因 tokens
        c: (B, hidden_dim)             图像条件向量
        """
        # 从图像条件生成 6 个调制参数
        params = self.adaLN_modulation(c)                        # (B, 6*hidden_dim)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            params.chunk(6, dim=-1)                              # 各 (B, hidden_dim)

        # 注意力分支：基因互相交流
        x_mod = modulate(self.norm1(x), shift_a, scale_a)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_a.unsqueeze(1) * attn_out                  # gate 控制注入量

        # FFN 分支：每个基因深度处理
        x_mod = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(x_mod)

        return x


# ─────────────────────────────────────────────────────────────
# 最终输出层
# ─────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """
    最后一层 adaLN + 线性投影
    把每个基因 token 的 hidden_dim 维向量压缩成 1 个标量（预测值）
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_dim, 1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        # 零初始化：训练开始时输出全为0，避免初期乱预测
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias,   0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_genes, hidden_dim)
        c: (B, hidden_dim)
        返回: (B, num_genes)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x).squeeze(-1)                        # (B, num_genes)


# ─────────────────────────────────────────────────────────────
# 邻居聚合（保持原样不动）
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
# 主模型
# ─────────────────────────────────────────────────────────────

class GenePredictor(nn.Module):
    def __init__(
        self,
        input_dim    : int   = 2048,
        hidden_dim   : int   = 256,    # DiT 隐层维度（推荐 128~256）
        num_layers   : int   = 4,      # DiT block 数量
        num_heads    : int   = 8,      # 注意力头数（需能整除 hidden_dim）
        output_dim   : int   = 200,    # 基因数
        dropout      : float = 0.1,
        use_neighbor : bool  = True,
        max_neighbors: int   = 6,
        gene_order   = None,
    ):
        super().__init__()
        self.output_dim   = output_dim
        self.hidden_dim   = hidden_dim
        self.use_neighbor = use_neighbor

        # 基因重排序（保持原有逻辑）
        if gene_order is None:
            gene_order = list(range(output_dim))
        _order = torch.tensor(gene_order, dtype=torch.long)
        self.register_buffer("gene_order",     _order)
        self.register_buffer("gene_order_inv", torch.argsort(_order))

        # ── 图像特征 → 条件向量 ───────────────────────────────────────────────
        # UNI+CONCH 2048维 → hidden_dim 维的条件向量
        self.img_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── 邻居聚合（可选）──────────────────────────────────────────────────
        if use_neighbor:
            self.neighbor_agg = NeighborAggregator(input_dim, hidden_dim, dropout)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # ── 基因身份嵌入 ──────────────────────────────────────────────────────
        # 200个基因，每个有自己的初始 token 向量
        self.gene_identity = GeneIdentityEmbedding(output_dim, hidden_dim)

        # ── DiT blocks ────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── 最终输出层 ────────────────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # 先统一初始化所有线性层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 再把 adaLN-Zero 相关层重新归零（覆盖上面的初始化）
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias,   0)

    def forward(self, img_emb, neighbor_zimg=None, neighbor_valid=None):
        B = img_emb.shape[0]

        # 1. 图像特征 → 条件向量 c
        c = self.img_proj(img_emb)                              # (B, hidden_dim)

        # 2. 邻居融合（如果有）
        if self.use_neighbor and neighbor_zimg is not None:
            nb = self.neighbor_agg(img_emb, neighbor_zimg, neighbor_valid)
            c  = self.fuse(torch.cat([c, nb], dim=-1))          # (B, hidden_dim)

        # 3. 初始化 200 个基因 tokens
        #    = 基因身份（"我是第几号基因"）+ 图像特征（"我在什么环境里"）
        #    这样每个 spot 的初始 token 是不同的，Transformer 能感知到 spot 差异
        img_token = c.unsqueeze(1)                              # (B, 1, hidden_dim)
        x = self.gene_identity(B) + img_token                  # (B, 200, hidden_dim)

        # 4. 过 DiT blocks
        #    每一层：基因之间做自注意力 + 被图像条件调制
        for block in self.blocks:
            x = block(x, c)                                     # (B, 200, hidden_dim)

        # 5. 输出基因预测值
        x0 = self.final_layer(x, c)                             # (B, 200)
        x0 = x0[:, self.gene_order_inv]

        # x_feat：取所有基因 token 的均值，用于 drift
        x_feat = x.mean(dim=1)                                  # (B, hidden_dim)

        return x0, x_feat


# 兼容旧版引用名
GeneMLP = GenePredictor