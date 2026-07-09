"""DriftST model components.

The model maps H&E-derived image embeddings to spatial gene expression. A
spot-level image representation is expanded into gene tokens through FiLM,
refined by self-attention over genes, and decoded into log1p expression. The
attention blocks can use a training-set co-expression matrix as a biological
bias and progressive gene gates to modulate residual updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborAggregator(nn.Module):
    """Attention pooling over spatial neighbors."""

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


class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioned on a spot-level image representation."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        nn.init.zeros_(self.ada_proj[-1].weight)
        nn.init.zeros_(self.ada_proj[-1].bias)

    def forward(self, x, c):
        scale_shift = self.ada_proj(c)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class GeneGate(nn.Module):
    """Context-aware gate controlling each gene token's residual update."""

    def __init__(self, hidden_dim, n_genes):
        super().__init__()
        self.token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.bio_bias = nn.Parameter(torch.zeros(n_genes))

    def init_from_degree(self, R, scale=0.1):
        """Initialize gate bias from absolute co-expression degree."""
        with torch.no_grad():
            degree = R.abs().sum(dim=1)
            degree = (degree - degree.mean()) / (degree.std() + 1e-6)
            self.bio_bias.copy_(degree * scale)

    def forward(self, gene_tokens):
        ctx = gene_tokens.mean(dim=1, keepdim=True)
        h = self.token_proj(gene_tokens) * self.context_proj(ctx)
        score = self.score(h).squeeze(-1) + self.bio_bias
        return torch.sigmoid(score).unsqueeze(-1)


class BioGuidedAttentionBlock(nn.Module):
    """Gene-dimension transformer block with co-expression attention bias."""

    def __init__(self, hidden_dim, num_heads, n_genes, dropout=0.1,
                 use_gate=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_gate = use_gate
        assert hidden_dim % num_heads == 0

        self.adanorm1 = AdaLN(hidden_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.bias_scale_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.bias_scale_proj[-1].weight)
        nn.init.constant_(self.bias_scale_proj[-1].bias, 0.1)

        self.attn_drop = nn.Dropout(dropout)
        self.adanorm2 = AdaLN(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        if use_gate:
            self.gene_gate = GeneGate(hidden_dim, n_genes)

    def forward(self, x, cond, bio_bias):
        B, N, D = x.shape

        gate = self.gene_gate(x) if self.use_gate else None
        x_norm = self.adanorm1(x, cond)

        Q = self.q_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias_scale = self.bias_scale_proj(cond)
        bio_bias_b = bio_bias.unsqueeze(0).unsqueeze(0)
        bio_term = bias_scale.view(B, self.num_heads, 1, 1) * bio_bias_b
        scores = scores + bio_term

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        x = x + gate * out if gate is not None else x + out
        ffn_out = self.ffn(self.adanorm2(x, cond))
        x = x + gate * ffn_out if gate is not None else x + ffn_out
        return x, gate


def gate_sparsity_loss(gates, target_fractions=None, entropy_weight=0.1):
    """Encourage layer-wise keep fractions and near-binary gates."""
    if not gates:
        return torch.tensor(0.0, device='cpu')

    n_layers = len(gates)
    if target_fractions is None:
        target_fractions = [
            0.85 - 0.35 * i / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]

    loss = torch.tensor(0.0, device=gates[0].device)
    for gate, target in zip(gates, target_fractions):
        loss = loss + (gate.mean() - target) ** 2

        g = gate.squeeze(-1).clamp(1e-6, 1 - 1e-6)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log()).mean()
        loss = loss + entropy_weight * entropy

    return loss


def _log_zinb_positive(x, mu, theta, pi_logits, eps=1e-8):
    """Element-wise ZINB log-likelihood following the standard scVI form."""
    if theta.dim() == 1:
        theta = theta.view(1, -1).expand_as(x)

    softplus_pi = F.softplus(-pi_logits)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi_logits + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = case_zero * (x < eps).float()

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = case_non_zero * (x > eps).float()

    return mul_case_zero + mul_case_non_zero


def zinb_loss(counts, x0, theta, pi_logits, eps=1e-8):
    """ZINB negative log-likelihood for raw counts.

    ``x0`` is predicted in log1p expression space and converted to count-scale
    NB means with ``mu = expm1(relu(x0))``.
    """
    mu = torch.expm1(F.relu(x0)).clamp_min(1e-4)
    ll = _log_zinb_positive(counts, mu, theta, pi_logits, eps)
    return -ll.mean()


class GenePredictor(nn.Module):
    """DriftST gene predictor."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        output_dim: int = 300,
        dropout: float = 0.1,
        use_neighbor: bool = False,
        max_neighbors: int = 6,
        n_attn_layers: int = 2,
        use_gate: bool = True,
        gate_target_fractions: list = None,
        gate_entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_neighbor = use_neighbor
        self.use_gate = use_gate
        self.gate_target_fractions = gate_target_fractions
        self.gate_entropy_weight = gate_entropy_weight

        self.register_buffer("bio_bias", torch.zeros(output_dim, output_dim))

        self.img_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.gene_emb = nn.Embedding(output_dim, hidden_dim)
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attn_blocks = nn.ModuleList([
            BioGuidedAttentionBlock(
                hidden_dim, num_heads, output_dim, dropout,
                use_gate=use_gate,
            )
            for _ in range(n_attn_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.gene_alpha = nn.Parameter(torch.ones(output_dim))
        self.gene_beta = nn.Parameter(torch.zeros(output_dim))

        self.zinb_pi_head = nn.Linear(hidden_dim, 1)
        self.zinb_log_theta = nn.Parameter(torch.zeros(output_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.ones_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def load_bio_bias(self, R):
        """Load the co-expression matrix and initialize gate priors."""
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

        c = self.img_proj(img_emb)
        if self.use_neighbor and neighbor_zimg is not None:
            nb = self.neighbor_agg(img_emb, neighbor_zimg, neighbor_valid)
            c = self.fuse(torch.cat([c, nb], dim=-1))
        else:
            c = self.fuse(c)

        h = c
        for block in self.encoder_blocks:
            h = h + block(h)

        gene_embs = self.gene_emb.weight.unsqueeze(0).expand(B, -1, -1)
        gamma = self.film_scale(gene_embs)
        beta = self.film_shift(gene_embs)

        h_expanded = h.unsqueeze(1).expand(-1, self.output_dim, -1)
        gene_tokens = self.token_proj(gamma * h_expanded + beta)

        all_gates = []
        for block in self.attn_blocks:
            gene_tokens, gate = block(gene_tokens, h, self.bio_bias)
            if gate is not None:
                all_gates.append(gate)

        h_out = self.out_norm(gene_tokens)
        x0 = self.head(h_out).squeeze(-1)
        x0 = self.gene_alpha * x0 + self.gene_beta

        model_out = {
            "zinb_pi_logits": self.zinb_pi_head(h_out).squeeze(-1),
            "zinb_theta": self.zinb_log_theta.exp().clamp(1e-4, 1e4),
        }
        if all_gates:
            model_out["gate_sparsity_loss"] = gate_sparsity_loss(
                all_gates, self.gate_target_fractions,
                entropy_weight=self.gate_entropy_weight,
            )

        return x0, model_out
