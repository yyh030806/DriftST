"""
model.py — DriftST: AdaLN + Bio-guided Attention + Progressive Gene Gating

架构：
  1. Encoder: 图像特征 → h (hidden representation)
  2. Gene Tokens: gene_emb 通过 FiLM 调制 h → 每个基因看到 h 的不同切面
  3. Bio-guided Self-Attention with AdaLN:
     - AdaLN: h 在每一层持续注入（不是开头加一次就丢掉）
     - attention scores += λ · R（共表达矩阵引导基因间交互）
     - Gene Gating: soft gate 控制残差，逐层剥离简单基因
  4. Output Head: 每个基因 → 标量预测

两个核心创新：
  (1) Bio-guided Attention: 已知基因调控关系 R 作为 attention bias
  (2) Progressive Gene Gating: 逐层剥离简单基因，深层聚焦难基因

接口：
  输入: img_emb, (neighbor_zimg, neighbor_valid 可选)
  输出: (x0, gate_info)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 邻居空间聚合（可选）
# ─────────────────────────────────────────────────────────────

class NeighborAggregator(nn.Module):
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
# AdaLN: Adaptive Layer Normalization
# ─────────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (DiT / GenAR 风格)。

    标准 LN: y = (x - μ) / σ * γ + β     （γ, β 是可学习参数）
    AdaLN:   y = (x - μ) / σ * (1 + γ_ada) + β_ada
             γ_ada, β_ada 由条件向量 c 动态生成

    image 信息 h 在每一层都参与调制，而不是只在开头加一次。
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # 从条件向量 c 生成 scale 和 shift
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        # 初始化为零 → 初始时等价于标准 LN（无 affine）
        nn.init.zeros_(self.ada_proj[-1].weight)
        nn.init.zeros_(self.ada_proj[-1].bias)

    def forward(self, x, c):
        """
        x: (B, N, D)  gene tokens
        c: (B, D)     spot-level conditioning（image representation h）
        """
        scale_shift = self.ada_proj(c)                 # (B, 2*D)
        scale, shift = scale_shift.chunk(2, dim=-1)    # 各 (B, D)
        scale = scale.unsqueeze(1)                     # (B, 1, D)
        shift = shift.unsqueeze(1)                     # (B, 1, D)
        return self.norm(x) * (1 + scale) + shift


# ─────────────────────────────────────────────────────────────
# Gene Gate: context-aware soft gating
# ─────────────────────────────────────────────────────────────

class GeneGate(nn.Module):
    """
    Context-aware soft gate，决定每个基因在当前层是否继续参与深层交互。

    gate 融合三个信号：
      1. 基因自身的 hidden state
      2. spot 全局上下文（所有基因 token 的均值池化）
      3. 生物网络先验偏置（hub 基因倾向留到深层）
    """

    def __init__(self, hidden_dim, n_genes):
        super().__init__()
        self.token_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score        = nn.Linear(hidden_dim, 1)
        self.bio_bias     = nn.Parameter(torch.zeros(n_genes))

    def init_from_degree(self, R, scale=0.1):
        """用共表达矩阵 degree 初始化。hub 基因 → 正偏置 → 留到深层。"""
        with torch.no_grad():
            degree = R.abs().sum(dim=1)
            degree = (degree - degree.mean()) / (degree.std() + 1e-6)
            self.bio_bias.copy_(degree * scale)

    def forward(self, gene_tokens):
        """
        gene_tokens: (B, N_genes, D)
        返回: gate (B, N_genes, 1)
        """
        ctx = gene_tokens.mean(dim=1, keepdim=True)
        h = self.token_proj(gene_tokens) * self.context_proj(ctx)
        score = self.score(h).squeeze(-1) + self.bio_bias
        return torch.sigmoid(score).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────
# Bio-guided Self-Attention Block (AdaLN + R bias + Gene Gating)
# ─────────────────────────────────────────────────────────────

class BioGuidedAttentionBlock(nn.Module):
    """
    Transformer block 集成三个特征：

    1. AdaLN: image representation h 在每层持续注入

    2. Bio-guided attention bias: scores += λ(h) · R
       λ 由 h 动态生成（spot-dependent），不同组织区域可以对 R 有不同信任程度
       例如模型可以学到"在肿瘤区域更信任共表达先验，正常区域少信任"

    3. Gene Gating: 逐层剥离简单基因
       gate 控制的是"谁还需要被更新"，不是"谁从序列中消失"。
       gate≈0 的基因自身 representation 冻结，但仍然作为 K/V 参与 attention，
       其携带的信息对其他基因仍可用。
    """

    def __init__(self, hidden_dim, num_heads, n_genes, dropout=0.1,
                 use_gate=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_gate = use_gate
        assert hidden_dim % num_heads == 0

        # AdaLN 替代标准 LN
        self.adanorm1 = AdaLN(hidden_dim, hidden_dim)

        # QKV
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 生物学 bias 缩放：由 h 动态决定每个 head 对 R 的信任程度
        # 不同 spot（肿瘤 vs 正常）可以学到不同的信任权重
        self.bias_scale_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        # 初始化：输出 ≈ 0.1（与原来的固定值一致）
        nn.init.zeros_(self.bias_scale_proj[-1].weight)
        nn.init.constant_(self.bias_scale_proj[-1].bias, 0.1)

        self.attn_drop = nn.Dropout(dropout)

        # FFN with AdaLN
        self.adanorm2 = AdaLN(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # Gene Gate
        if use_gate:
            self.gene_gate = GeneGate(hidden_dim, n_genes)

    def forward(self, x, cond, bio_bias):
        """
        x:        (B, N_genes, hidden_dim)   gene tokens
        cond:     (B, hidden_dim)            spot-level conditioning (h)
        bio_bias: (N_genes, N_genes)         全局共表达矩阵
        """
        B, N, D = x.shape

        # ── Gate ─────────────────────────────────────────────
        gate = None
        if self.use_gate:
            gate = self.gene_gate(x)

        # ── Self-Attention with AdaLN + bio bias ─────────────
        x_norm = self.adanorm1(x, cond)

        Q = self.q_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 生物学先验：λ 由 h 动态决定（spot-dependent）
        bias_scale = self.bias_scale_proj(cond)                # (B, num_heads)
        bio_bias_b = bio_bias.unsqueeze(0).unsqueeze(0)        # (1, 1, G, G)
        bio_term   = bias_scale.view(B, self.num_heads, 1, 1) * bio_bias_b
        scores     = scores + bio_term

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        # ── Gated residual ───────────────────────────────────
        if gate is not None:
            x = x + gate * out
        else:
            x = x + out

        # ── FFN ──────────────────────────────────────────────
        ffn_out = self.ffn(self.adanorm2(x, cond))
        if gate is not None:
            x = x + gate * ffn_out
        else:
            x = x + ffn_out

        return x, gate


# ─────────────────────────────────────────────────────────────
# Gate Sparsity Loss
# ─────────────────────────────────────────────────────────────

def gate_sparsity_loss(gates, target_fractions=None, entropy_weight=0.1):
    """
    两项正则共同约束 gate：

    1. 均值约束：控制每层的保留比例，形成漏斗
       L_frac = (mean(gate) - target)²

    2. 熵正则：鼓励 gate 走向 0 或 1（二值化），避免所有基因"半开半关"
       L_ent = -mean(g·log(g) + (1-g)·log(1-g))
       熵最大 = 0.693（g=0.5），最小 = 0（g=0 或 1）
       最小化熵 → 推向二值化
    """
    if not gates:
        return torch.tensor(0.0, device=gates[0].device if gates else 'cpu')

    n_layers = len(gates)
    if target_fractions is None:
        target_fractions = [
            0.85 - 0.35 * i / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]

    loss = torch.tensor(0.0, device=gates[0].device)
    for gate, target in zip(gates, target_fractions):
        # 均值约束
        loss = loss + (gate.mean() - target) ** 2

        # 熵正则：推向二值化
        g = gate.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log()).mean()
        loss = loss + entropy_weight * entropy

    return loss


# ─────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────

class GenePredictor(nn.Module):
    """
    DriftST Gene Predictor

    信息流：
      Image Features (2048d)
            │
        Encoder (残差 MLP)
            │
         h (hidden_dim)     ──────────────────────┐
            │                                      │
      FiLM: gene_emb 调制 h                        │ 每层 AdaLN 持续注入
            │                                      │
      Bio-guided Attention × N layers  ←───────────┘
        · AdaLN conditioning
        · scores += λ · R
        · gated residual
            │
      LN → Linear → scalar per gene
            │
      Prediction (B, 300)
    """

    def __init__(
        self,
        input_dim:       int   = 2048,
        hidden_dim:      int   = 256,
        num_layers:      int   = 2,
        num_heads:       int   = 8,
        output_dim:      int   = 300,
        dropout:         float = 0.1,
        use_neighbor:    bool  = False,
        max_neighbors:   int   = 6,
        n_attn_layers:   int   = 2,
        use_gate:        bool  = True,
        gate_target_fractions: list = None,
        gate_entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.output_dim    = output_dim
        self.hidden_dim    = hidden_dim
        self.use_neighbor  = use_neighbor
        self.use_gate      = use_gate
        self.gate_target_fractions = gate_target_fractions
        self.gate_entropy_weight   = gate_entropy_weight

        # ── 生物学先验矩阵 ───────────────────────────────────
        # 全局基因共表达矩阵（从训练集表达矩阵自身计算）
        self.register_buffer(
            "bio_bias", torch.zeros(output_dim, output_dim)
        )

        # ── 图像特征投影 ─────────────────────────────────────
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

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Encoder: 残差 MLP ────────────────────────────────
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # ── Gene Embeddings ──────────────────────────────────
        self.gene_emb = nn.Embedding(output_dim, hidden_dim)

        # ── FiLM: gene_emb → scale, shift 调制 h ────────────
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)

        # gene token 投影
        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Bio-guided Attention layers ──────────────────────
        self.attn_blocks = nn.ModuleList([
            BioGuidedAttentionBlock(
                hidden_dim, num_heads, output_dim, dropout,
                use_gate=use_gate,
            )
            for _ in range(n_attn_layers)
        ])

        # ── 输出头 ───────────────────────────────────────────
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        # FiLM: 初始 scale ≈ 1, shift ≈ 0 → 初始时 gene_tokens ≈ h（接近原始行为）
        nn.init.zeros_(self.film_scale.weight)
        nn.init.ones_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def load_bio_bias(self, R):
        """加载全局基因共表达矩阵，并初始化 gate 的 bio_bias。"""
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32)
        assert R.shape == (self.output_dim, self.output_dim)
        self.bio_bias.copy_(R)

        if self.use_gate:
            for block in self.attn_blocks:
                if hasattr(block, 'gene_gate'):
                    block.gene_gate.init_from_degree(R, scale=0.1)

    def forward(self, img_emb, neighbor_zimg=None, neighbor_valid=None):
        B = img_emb.shape[0]

        # ── Encoder ──────────────────────────────────────────
        c = self.img_proj(img_emb)

        if self.use_neighbor and neighbor_zimg is not None:
            nb = self.neighbor_agg(img_emb, neighbor_zimg, neighbor_valid)
            c = self.fuse(torch.cat([c, nb], dim=-1))
        else:
            c = self.fuse(c)

        h = c
        for block in self.encoder_blocks:
            h = h + block(h)                                    # (B, hidden)

        # ── 生物学先验：全局基因共表达矩阵 ────────────────────
        bio_bias = self.bio_bias                                # (G, G)

        # ── Gene Tokens via FiLM ─────────────────────────────
        gene_embs = self.gene_emb.weight.unsqueeze(0).expand(B, -1, -1)
        gamma = self.film_scale(gene_embs)                     # (B, G, hidden)
        beta  = self.film_shift(gene_embs)                     # (B, G, hidden)

        h_expanded = h.unsqueeze(1).expand(-1, self.output_dim, -1)
        gene_tokens = self.token_proj(gamma * h_expanded + beta)

        # ── Bio-guided Attention (AdaLN + R_effective + Gating) ──
        all_gates = []
        for block in self.attn_blocks:
            gene_tokens, gate = block(gene_tokens, h, bio_bias)
            if gate is not None:
                all_gates.append(gate)

        # ── Output ───────────────────────────────────────────
        x0 = self.head(self.out_norm(gene_tokens)).squeeze(-1)

        # ── model_out：gate 信息打包 ────────────────────────
        model_out = {}
        if all_gates:
            model_out['gate_sparsity_loss'] = gate_sparsity_loss(
                all_gates, self.gate_target_fractions,
                entropy_weight=self.gate_entropy_weight,
            )

        return x0, model_out or None