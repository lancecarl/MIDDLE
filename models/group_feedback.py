"""Group-risk feedback modules for Phase 3 inference."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


class GroupRiskHead(nn.Module):
    """GroupRiskHead module."""

    def __init__(self, coll_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coll_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, H_C: torch.Tensor) -> torch.Tensor:
        """Run forward."""
        return torch.sigmoid(self.mlp(H_C)).squeeze(-1)   # [K]


class GroupFeedback(nn.Module):
    """GroupFeedback module."""

    def __init__(
        self,
        coll_dim: int = 128,
        K: int = 10,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.coll_dim   = coll_dim
        self.K          = K
        self.output_dim = output_dim


        self.group_risk_head = GroupRiskHead(coll_dim, hidden_dim=64)


        attn_in_dim = coll_dim + 3
        self.attn_net = nn.Sequential(
            nn.Linear(attn_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )


        self.output_proj = nn.Sequential(
            nn.Linear(coll_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )


        self._fallback: Optional[torch.Tensor] = None


    def compute_group_risk(
        self,
        H_C: torch.Tensor,    # [K, coll_dim]
        S: torch.Tensor,      # [U_global, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run compute_group_risk."""
        rho_k = self.group_risk_head(H_C)         # [K]
        # c_u^group = Σ_k S_{u,k} × ρ_k
        c_u_group = (S * rho_k.unsqueeze(0)).sum(-1)
        return rho_k, c_u_group


    def _get_fallback(self, device: torch.device) -> torch.Tensor:
        """Run _get_fallback."""
        if self._fallback is None or self._fallback.device != device:
            self._fallback = torch.zeros(self.output_dim, device=device)
        return self._fallback

    def forward(
        self,
        H_C: torch.Tensor,                # [K, coll_dim]
        c_u_ind: torch.Tensor,
        S: torch.Tensor,                  # [U_global, K]
        batch_data: Batch,
        global_user_emb: torch.Tensor,
        twitter_id_to_global_idx: Dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward."""
        device = H_C.device


        rho_k, c_u_group = self.compute_group_risk(H_C, S)   # [K], [U_global]

        B = batch_data.num_graphs
        h_group_list: List[torch.Tensor] = []


        twitter_ids_all: torch.Tensor = getattr(batch_data, 'twitter_ids', None)
        batch_assign: torch.Tensor = batch_data.batch         # [N_total]

        for graph_idx in range(B):
            node_mask = (batch_assign == graph_idx)           # [N_total]

            if twitter_ids_all is not None:

                tids = twitter_ids_all[node_mask].cpu().tolist()  # List[int]
            else:
                tids = []


            valid_indices = []
            valid_pos     = []
            for local_pos, tid in enumerate(tids):
                if tid in twitter_id_to_global_idx and tid != -1:
                    valid_indices.append(twitter_id_to_global_idx[tid])
                    valid_pos.append(local_pos / max(len(tids) - 1, 1))

            if len(valid_indices) == 0:

                h_group_list.append(self._get_fallback(device))
                continue

            idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
            pos_tensor = torch.tensor(valid_pos, dtype=torch.float32, device=device)


            H_u       = global_user_emb[idx_tensor]      # [M, coll_dim]
            c_ind_u   = c_u_ind[idx_tensor]              # [M]
            c_grp_u   = c_u_group[idx_tensor]            # [M]


            attn_in = torch.cat([
                H_u,
                c_ind_u.unsqueeze(-1),
                c_grp_u.unsqueeze(-1),
                pos_tensor.unsqueeze(-1),
            ], dim=-1)                                   # [M, D+3]

            attn_scores = self.attn_net(attn_in).squeeze(-1)   # [M]
            gamma       = F.softmax(attn_scores, dim=0)        # [M]


            H_agg    = (gamma.unsqueeze(-1) * H_u).sum(0)     # [coll_dim]
            c_grp_agg = (gamma * c_grp_u).sum(0, keepdim=True)  # [1]

            combined = torch.cat([H_agg, c_grp_agg], dim=-1)  # [coll_dim+1]
            h_i_group = self.output_proj(combined)             # [output_dim]
            h_group_list.append(h_i_group)

        h_group = torch.stack(h_group_list, dim=0)   # [B, output_dim]
        return h_group, c_u_group
