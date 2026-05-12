"""
Risk-modulated message passing.

RiskModulatedConv gates messages by user risk scores and aggregates them
into graph-level features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import global_add_pool
from typing import Tuple


# Helper: risk-weighted global pooling

def risk_weighted_global_pool(
    x: torch.Tensor,
    r_u: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Run risk_weighted_global_pool."""
    w = (1.0 - r_u).clamp(min=1e-6).unsqueeze(-1)   # [N, 1]
    x_weighted = x * w                                 # [N, D]


    denom = global_add_pool(w, batch, size=num_graphs)  # [B, 1]
    denom = denom.clamp(min=1e-6)

    numer = global_add_pool(x_weighted, batch, size=num_graphs)  # [B, D]
    return numer / denom


# Core module: risk-modulated message passing

class RiskModulatedConv(nn.Module):
    """RiskModulatedConv module."""

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        risk_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim

        self.risk_embed = nn.Sequential(
            nn.Linear(1, risk_dim),
            nn.ReLU(),
            nn.Linear(risk_dim, risk_dim),
        )

        gate_in_dim = input_dim + risk_dim + input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )


        self.diff_proj = nn.Linear(input_dim, input_dim)


        node_update_in = input_dim * 3 + risk_dim
        self.node_update = nn.Sequential(
            nn.Linear(node_update_in, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )


        self.residual = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        r_u: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        N = h.shape[0]
        device = h.device

        # ── 0. Empty-edge fallback ─────────────────────────────────────────
        if edge_index.shape[1] == 0:
            node_feat = self.node_update(
                torch.cat([
                    h,
                    torch.zeros(N, self.input_dim, device=device),
                    torch.zeros(N, self.input_dim, device=device),
                    self.risk_embed(r_u.unsqueeze(-1)),
                ], dim=-1)
            ) + self.residual(h)
            graph_feat = risk_weighted_global_pool(node_feat, r_u, batch, num_graphs)
            return graph_feat, node_feat

        src, dst = edge_index[0], edge_index[1]   # direction: src -> dst

        # ── 1. Risk embedding ──────────────────────────────────────────────
        r_vec = self.risk_embed(r_u.unsqueeze(-1))   # [N, risk_dim]

        # ── 2. Edge gate gate_{uv} ─────────────────────────────────────────
        gate_input = torch.cat([
            h[src],                   # sender features  [E, input_dim]
            r_vec[src],               # sender risk vec  [E, risk_dim]
            h[dst],                   # receiver feats   [E, input_dim]
        ], dim=-1)                                     # [E, gate_in_dim]

        gate = torch.sigmoid(self.gate_net(gate_input))   # [E, input_dim]

        # ── 3. Gated message: msg = gate ⊙ h_src ───────────────────────────
        msg = gate * h[src]                               # [E, input_dim]

        # ── 4. Risk-adaptive aggregation ───────────────────────────────────
        trust_weight = (1.0 - r_u[src]).clamp(min=1e-6)  # [E]
        alpha = pyg_softmax(trust_weight, dst, num_nodes=N)   # [E], softmax per dst

        weighted_msg = alpha.unsqueeze(-1) * msg          # [E, input_dim]

        # Aggregate -> agg_msg [N, input_dim]
        agg_msg = torch.zeros(N, self.input_dim, device=device)
        agg_msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

        # ── 5. Difference aggregation (h_src - h_dst) ─────────────────────
        diff = h[src] - h[dst]                            # [E, input_dim]
        diff_weighted = alpha.unsqueeze(-1) * self.diff_proj(diff)
        diff_agg = torch.zeros(N, self.input_dim, device=device)
        diff_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(diff_weighted), diff_weighted)

        # ── 6. Node feature update ────────────────────────────────────────
        node_update_input = torch.cat([h, agg_msg, diff_agg, r_vec], dim=-1)
        node_feat = self.node_update(node_update_input) + self.residual(h)
        node_feat = self.dropout(node_feat)                # [N, output_dim]

        # ── 7. Graph-level risk-weighted pooling ──────────────────────────
        graph_feat = risk_weighted_global_pool(
            node_feat, r_u, batch, num_graphs
        )                                                  # [B, output_dim]

        return graph_feat, node_feat
