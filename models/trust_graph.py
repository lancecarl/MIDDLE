"""Trust and user-risk encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class ImprovedTrustEncoder(nn.Module):
    """ImprovedTrustEncoder module."""

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128,
                 num_heads=8, dropout=0.3, use_residual=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual


        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=None,
            add_self_loops=True,
            share_weights=False
        )


        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=output_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=None,
            add_self_loops=True
        )


        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(output_dim)


        if self.use_residual:
            self.residual1 = nn.Linear(input_dim, hidden_dim)
            self.residual2 = nn.Linear(hidden_dim, output_dim)

        # ── Dropout ──
        self.dropout = nn.Dropout(dropout)


        self.pool_projection = nn.Linear(output_dim * 2, output_dim)

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def _node_forward(self, x, edge_index):
        """Run _node_forward."""
        identity = x


        h = self.gat1(x, edge_index)
        if self.use_residual:
            h = h + self.residual1(identity)
        h = self.ln1(h)
        h = F.elu(h)
        h = self.dropout(h)


        identity2 = h
        h = self.gat2(h, edge_index)
        if self.use_residual:
            h = h + self.residual2(identity2)
        h = self.ln2(h)
        h = F.elu(h)

        return h  # [N, output_dim]

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def forward(self, x, edge_index, batch, return_node_embeddings=False):
        """Run forward."""
        h = self._node_forward(x, edge_index)   # [N, output_dim]


        if return_node_embeddings:
            return h


        h_mean = global_mean_pool(h, batch)              # [B, output_dim]
        h_max  = global_max_pool(h, batch)               # [B, output_dim]
        h_combined = torch.cat([h_mean, h_max], dim=1)  # [B, output_dim*2]
        graph_emb  = self.pool_projection(h_combined)   # [B, output_dim]

        return graph_emb

    def get_node_embeddings(self, x, edge_index):
        """Run get_node_embeddings."""
        return self._node_forward(x, edge_index)

    def get_attention_weights(self, x, edge_index):
        """Run get_attention_weights."""
        h, (edge_index_out, alpha) = self.gat1(
            x, edge_index, return_attention_weights=True
        )
        return edge_index_out, alpha


class UserRiskMILEncoder(nn.Module):
    """UserRiskMILEncoder module."""

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128,
                 num_heads=8, dropout=0.3):
        """Run __init__."""
        super().__init__()


        self.encoder = ImprovedTrustEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_residual=True
        )


        self.risk_head = nn.Linear(output_dim, 1)


        self.attention_net = nn.Sequential(
            nn.Linear(output_dim + 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mil_product_stable(pi_r: torch.Tensor) -> torch.Tensor:
        """Run _mil_product_stable."""
        eps = 1e-7

        pi_r_clipped = pi_r.clamp(0.0, 1.0 - eps)
        log_complement = torch.log(1.0 - pi_r_clipped + eps)
        p = 1.0 - torch.exp(log_complement.sum())
        return p.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    def forward(self, x, edge_index, batch, node_depth=None):
        """Run forward."""

        z_u = self.encoder(x, edge_index, batch,
                           return_node_embeddings=True)  # [N, output_dim]


        r_u = torch.sigmoid(self.risk_head(z_u)).squeeze(-1)   # [N]


        if node_depth is not None:
            depth_norm   = (node_depth.float() / 10.0).unsqueeze(-1)   # [N,1]
            is_non_root  = (node_depth > 0).float().unsqueeze(-1)       # [N,1]
        else:
            depth_norm   = torch.zeros(x.shape[0], 1, device=x.device)
            is_non_root  = torch.ones(x.shape[0], 1, device=x.device)

        attn_input = torch.cat([z_u, depth_norm, is_non_root], dim=-1)  # [N, D+2]
        attn_raw   = self.attention_net(attn_input).squeeze(-1)          # [N]


        num_graphs = int(batch.max().item()) + 1
        p_user_list = []

        for graph_idx in range(num_graphs):
            node_mask = (batch == graph_idx)          # [N] bool


            if node_depth is not None:
                user_mask = node_mask & (node_depth > 0)
            else:

                indices = node_mask.nonzero(as_tuple=True)[0]
                if len(indices) > 1:
                    user_mask = node_mask.clone()
                    user_mask[indices[0]] = False
                else:
                    user_mask = node_mask

            r_masked    = r_u[user_mask]            # [M]
            pi_raw      = attn_raw[user_mask]        # [M]

            if r_masked.numel() == 0:

                p_user_list.append(torch.zeros(1, device=x.device).squeeze())
                continue


            pi = F.softmax(pi_raw, dim=0)            # [M]，Σπ=1


            pi_r = pi * r_masked
            p_user = self._mil_product_stable(pi_r)
            p_user_list.append(p_user)

        p_user_per_graph = torch.stack(p_user_list)  # [B]

        return z_u, r_u, p_user_per_graph


class HierarchicalTrustEncoder(nn.Module):
    """HierarchicalTrustEncoder module."""

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128,
                 num_heads=8, dropout=0.3):
        super().__init__()

        self.local_encoder = GATv2Conv(
            input_dim, hidden_dim, heads=num_heads, concat=False
        )
        self.global_encoder = GATv2Conv(
            hidden_dim, output_dim, heads=num_heads, concat=False
        )

        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        """Run forward."""
        h_local = self.local_encoder(x, edge_index)
        h_local = F.elu(h_local)
        h_local = self.dropout(h_local)

        h_global = self.global_encoder(h_local, edge_index)
        h_global = F.elu(h_global)

        local_pool  = global_mean_pool(h_local,  batch)
        global_pool = global_mean_pool(h_global, batch)

        hierarchical = torch.cat([local_pool, global_pool], dim=1)
        fused = self.hierarchical_fusion(hierarchical)

        return fused

    def get_node_embeddings(self, x, edge_index):
        """Run get_node_embeddings."""
        h_local  = self.local_encoder(x, edge_index)
        h_local  = F.elu(h_local)
        h_global = self.global_encoder(h_local, edge_index)
        h_global = F.elu(h_global)
        return h_global
