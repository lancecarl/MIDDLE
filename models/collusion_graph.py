"""Learnable user collusion graph modules."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class CollusionGraphBuilder:
    """CollusionGraphBuilder module."""

    def __init__(self, num_events: int):
        self.num_events = num_events

    @staticmethod
    def _sync_weight(t_u: float, t_v: float, eps: float = 3600.0) -> float:
        """sync(u,v,i) = exp(-|τ_{u,i} - τ_{v,i}| / ε)"""
        return math.exp(-abs(t_u - t_v) / (eps + 1e-6))

    def build(
        self,
        B: Dict[int, Dict[int, float]],          # {user_id: {event_idx: timestamp}}
        p_hat: Dict[int, float],
        idf: torch.Tensor,                        # [N_events]
        min_cooccur: int = 2,
        eps: float = 3600.0,
        p_threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run build."""
        users = list(B.keys())
        U = len(users)
        uid2idx = {u: i for i, u in enumerate(users)}


        idf_np = idf.cpu().numpy() if isinstance(idf, torch.Tensor) else idf

        edges: List[Tuple[int, int, float]] = []  # (i, j, weight)

        for a in range(U):
            ua = users[a]
            events_a = B[ua]
            for b in range(a + 1, U):
                ub = users[b]
                events_b = B[ub]


                common = set(events_a.keys()) & set(events_b.keys())
                if len(common) < min_cooccur:
                    continue

                w = 0.0
                for ei in common:
                    p_i = p_hat.get(ei, 0.0)
                    if p_i < p_threshold:
                        continue
                    idf_i = float(idf_np[ei]) if ei < len(idf_np) else 1.0
                    t_ua  = events_a[ei]
                    t_ub  = events_b[ei]
                    sync  = self._sync_weight(t_ua, t_ub, eps)
                    w    += p_i * idf_i * sync

                if w > 0:
                    edges.append((uid2idx[ua], uid2idx[ub], w))

        if not edges:
            empty_ei = torch.zeros((2, 0), dtype=torch.long)
            empty_ew = torch.zeros(0, dtype=torch.float)
            return empty_ei, empty_ew


        src_list, dst_list, w_list = [], [], []
        for i, j, w in edges:
            src_list += [i, j]
            dst_list += [j, i]
            w_list   += [w, w]

        edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(w_list, dtype=torch.float)


        max_w = edge_weight.max().clamp(min=1e-6)
        edge_weight = edge_weight / max_w

        return edge_index, edge_weight


class LearnableSparseGate(nn.Module):
    """LearnableSparseGate module."""

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.gate_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,          # [U, node_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_weight: torch.Tensor, # [E]
    ) -> torch.Tensor:
        """
        Returns:
            M_uv: [E] ∈ [0,1]
        """
        if edge_index.shape[1] == 0:
            return torch.zeros(0, device=h.device)

        src, dst = edge_index[0], edge_index[1]
        gate_in  = torch.cat([
            h[src],
            h[dst],
            edge_weight.unsqueeze(-1),
        ], dim=-1)                              # [E, 2*node_dim+1]
        M_uv = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)   # [E]
        return M_uv


class SoftCommunityAssign(nn.Module):
    """SoftCommunityAssign module."""

    def __init__(self, node_dim: int, K: int = 10, temperature: float = 1.0):
        super().__init__()
        self.K = K
        self.temperature = temperature

        self.centroids = nn.Parameter(torch.randn(K, node_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Run forward."""

        diff = h.unsqueeze(1) - self.centroids.unsqueeze(0)   # [U, K, D]
        dist = (diff ** 2).sum(-1)                             # [U, K]
        S = F.softmax(-dist / self.temperature, dim=-1)        # [U, K]
        return S

    def get_cluster_features(
        self, h: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """Run get_cluster_features."""
        # H_C[k] = Σ_u S[u,k] * h[u] / Σ_u S[u,k]
        S_T = S.t()                                    # [K, U]
        denom = S_T.sum(-1, keepdim=True).clamp(1e-6)
        H_C = S_T @ h / denom                         # [K, D]
        return H_C


class CollusionGNN(nn.Module):
    """CollusionGNN module."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=output_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=True,
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)


        self.res1 = nn.Linear(input_dim, hidden_dim)
        self.res2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        h: torch.Tensor,           # [U, input_dim]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr: torch.Tensor,    # [E] — M_uv * w_uv
    ) -> torch.Tensor:
        """Returns: node_emb [U, output_dim]"""
        if edge_index.shape[1] == 0:

            h1 = F.elu(self.res1(h))
            h2 = F.elu(self.res2(h1))
            return self.ln2(h2)

        ea = edge_attr.unsqueeze(-1)   # [E, 1]


        h1 = self.gat1(h, edge_index, ea) + self.res1(h)
        h1 = self.ln1(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)


        h2 = self.gat2(h1, edge_index, ea) + self.res2(h1)
        h2 = self.ln2(h2)
        h2 = F.elu(h2)

        return h2   # [U, output_dim]


class ConfidenceScorer(nn.Module):
    """ConfidenceScorer module."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns: c_u [U]"""
        return torch.sigmoid(self.scorer(h)).squeeze(-1)


class CollusionGraphModule(nn.Module):
    """CollusionGraphModule module."""

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        K: int = 10,
        num_heads: int = 4,
        dropout: float = 0.3,
        temperature: float = 1.0,
        input_dim: Optional[int] = None,
    ):
        super().__init__()

        self.node_dim    = node_dim
        self.output_dim  = output_dim
        self.K           = K
        # feature_dim: the *actual* incoming z_u dimension.
        # When trainer collects raw BERT features (768-d) instead of the
        # encoder output (node_dim-d), input_dim should be set to 768.
        # A linear projection maps it down to node_dim before the gate/GNN.
        self.feature_dim = input_dim if input_dim is not None else node_dim

        # Input projection: only added when the stored feature dim differs
        # from the internal node_dim (e.g. raw BERT 768 vs GNN hidden 128).
        # Loaded with strict=False so missing keys fall back to this
        # random-init layer gracefully.
        if self.feature_dim != self.node_dim:
            self.input_proj: Optional[nn.Linear] = nn.Linear(
                self.feature_dim, self.node_dim, bias=False
            )
        else:
            self.input_proj = None


        self.sparse_gate = LearnableSparseGate(node_dim, hidden_dim=64)


        self.gnn = CollusionGNN(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )


        self.community_assign = SoftCommunityAssign(
            node_dim=output_dim, K=K, temperature=temperature
        )


        self.conf_scorer = ConfidenceScorer(output_dim)


    def forward(
        self,
        z_u: torch.Tensor,           # [U, node_dim]
        edge_index: torch.Tensor,     # [2, E]
        edge_weight: torch.Tensor,    # [E]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            H_C        : [K, output_dim]
            c_u_scores : [U]
            S          : [U, K]
            M_uv       : [E]
        """


        if self.input_proj is not None:
            z_u = self.input_proj(z_u)          # [U, feature_dim] → [U, node_dim]
        elif z_u.shape[-1] != self.node_dim:


            import warnings
            warnings.warn(
                f"[CollusionGraphModule] z_u dim {z_u.shape[-1]} != "
                f"node_dim {self.node_dim}. "
                "Rebuild collusion_data or pass input_dim= at init for a trained projection. "
                "Falling back to a random orthogonal projection.",
                stacklevel=2,
            )
            proj = nn.Linear(z_u.shape[-1], self.node_dim, bias=False).to(z_u.device)
            nn.init.orthogonal_(proj.weight)
            with torch.no_grad():
                z_u = proj(z_u)


        M_uv = self.sparse_gate(z_u, edge_index, edge_weight)   # [E]


        effective_weight = M_uv * edge_weight


        h_out = self.gnn(z_u, edge_index, effective_weight)      # [U, output_dim]


        S   = self.community_assign(h_out)                       # [U, K]
        H_C = self.community_assign.get_cluster_features(h_out, S)  # [K, output_dim]


        c_u_scores = self.conf_scorer(h_out)                     # [U]

        return H_C, c_u_scores, S, M_uv


    @staticmethod
    def compute_idf(
        B: Dict[int, Dict[int, float]],
        num_events: int,
    ) -> torch.Tensor:
        """Run compute_idf."""
        U = len(B)
        count = torch.zeros(num_events)
        for events in B.values():
            for ei in events:
                if 0 <= ei < num_events:
                    count[ei] += 1.0
        idf = torch.log((U + 1.0) / (1.0 + count))
        return idf
