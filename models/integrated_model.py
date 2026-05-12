"""Integrated MR-DE-TGM model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


from .temporal_propagation import ImprovedTemporalPropagationEncoder
from .trust_graph import UserRiskMILEncoder
from .risk_modulated_conv import RiskModulatedConv, risk_weighted_global_pool
from .collusion_graph import CollusionGraphModule
from .group_feedback import GroupFeedback


class InitClassifier(nn.Module):
    """InitClassifier module."""
    def __init__(self, prop_dim: int, user_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prop_dim + user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_prop: torch.Tensor, z_u_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prop     : [B, prop_dim]
            z_u_pooled : [B, user_dim]
        Returns:
            p_hat_0 : [B]
        """
        feat = torch.cat([h_prop, z_u_pooled], dim=-1)   # [B, prop_dim+user_dim]
        return torch.sigmoid(self.net(feat)).squeeze(-1)   # [B]


class FinalClassifier(nn.Module):
    """Final event classifier for propagation, user, and group features."""

    def __init__(
        self,
        prop_dim: int,
        user_dim: int,
        group_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        in_dim = prop_dim + user_dim + group_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h_prop_mod: torch.Tensor,
        z_u_pooled: torch.Tensor,
        h_group: torch.Tensor,
    ) -> torch.Tensor:
        """Return event risk probabilities with shape [B]."""
        feat = torch.cat([h_prop_mod, z_u_pooled, h_group], dim=-1)
        return torch.sigmoid(self.net(feat)).squeeze(-1)


class MRDETGMModel(nn.Module):
    """MRDETGMModel module."""

    def __init__(
        self,
        config,
        fusion_type: str = 'attention',
    ):
        super().__init__()


        if isinstance(config, dict):
            node_dim      = config.get('node_dim', 768)
            prop_hidden   = config.get('prop_hidden', 256)
            prop_dim      = config.get('prop_dim', 128)
            trust_hidden  = config.get('trust_hidden', 256)
            trust_dim     = config.get('trust_dim', 128)
            fusion_hidden = config.get('fusion_hidden', 256)
            num_heads     = config.get('num_heads', 8)
            window_size   = config.get('window_size', 3600)
            K             = config.get('collusion_k', 10)
            coll_hidden   = config.get('collusion_hidden', 128)
            group_out_dim = config.get('group_dim', 64)
        else:
            node_dim      = getattr(config, 'NODE_DIM', 768)
            prop_hidden   = getattr(config, 'PROP_HIDDEN', 256)
            prop_dim      = getattr(config, 'PROP_DIM', 128)
            trust_hidden  = getattr(config, 'TRUST_HIDDEN', 256)
            trust_dim     = getattr(config, 'TRUST_DIM', 128)
            fusion_hidden = getattr(config, 'FUSION_HIDDEN', 256)
            num_heads     = getattr(config, 'NUM_HEADS', 8)
            window_size   = getattr(config, 'WINDOW_SIZE', 3600)
            K             = getattr(config, 'COLLUSION_K', 10)
            coll_hidden   = getattr(config, 'COLLUSION_HIDDEN', 128)
            group_out_dim = getattr(config, 'GROUP_DIM', 64)

        self.prop_dim      = prop_dim
        self.user_dim      = trust_dim
        self.coll_dim      = coll_hidden
        self.group_dim     = group_out_dim
        self.K             = K
        self._use_phase2   = False


        self.prop_encoder = ImprovedTemporalPropagationEncoder(
            input_dim=node_dim,
            hidden_dim=prop_hidden,
            output_dim=prop_dim,
            gcn_dim=64,
            transformer_hidden=192,
            window_size=window_size,
        )


        self.user_risk_encoder = UserRiskMILEncoder(
            input_dim=node_dim,
            hidden_dim=trust_hidden,
            output_dim=trust_dim,
            num_heads=num_heads,
            dropout=0.3,
        )


        self.risk_modulated_conv = RiskModulatedConv(
            input_dim=node_dim,
            output_dim=prop_dim,
        )


        self.init_classifier = InitClassifier(prop_dim, trust_dim, hidden_dim=fusion_hidden)


        self.final_classifier = FinalClassifier(
            prop_dim=prop_dim,
            user_dim=trust_dim,
            group_dim=group_out_dim,
            hidden_dim=fusion_hidden,
            dropout=0.4,
        )


        self.collusion_graph: Optional[CollusionGraphModule] = None


        self.group_feedback: Optional[GroupFeedback] = None


        self._twitter_id_to_global_idx: Optional[Dict[int, int]] = None

        print(f"\nMRDETGMModel initialized:")
        print(f"  - propagation encoder: Stage-aware Transformer ({prop_dim}D)")
        print(f"  - user risk encoder: UserRiskMIL ({trust_dim}D)")
        print(f"  - risk-modulated propagation: RiskModulatedConv")
        print(f"  - Phase 2 modules: {'enabled' if self._use_phase2 else 'disabled (Phase 1 mode)'}")


    def set_phase2_modules(
        self,
        collusion_graph: CollusionGraphModule,
        group_feedback: GroupFeedback,
        twitter_id_to_global_idx: Dict[int, int],
    ) -> None:
        """Attach Phase 2 modules after the collusion graph is built."""
        self.collusion_graph = collusion_graph
        self.group_feedback = group_feedback
        self._twitter_id_to_global_idx = twitter_id_to_global_idx
        self._use_phase2 = True
        print(f"Phase 2 modules enabled (collusion K={self.K})")

    def initialize_phase2_modules(
        self,
        config,
        device: torch.device,
        twitter_id_to_global_idx: Optional[Dict[int, int]] = None,
    ) -> None:
        """Run initialize_phase2_modules."""
        if self._use_phase2:
            print("Phase 2 modules already initialized; skipping.")
            return

        def _get(key, default):
            if hasattr(config, key):
                return getattr(config, key)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        coll_hidden   = _get('COLLUSION_HIDDEN', 128)
        coll_k        = _get('COLLUSION_K', 10)
        group_out_dim = _get('GROUP_DIM', 64)

        collusion_mod = CollusionGraphModule(
            node_dim=coll_hidden,
            hidden_dim=coll_hidden,
            output_dim=coll_hidden,
            K=coll_k,
        ).to(device)

        group_fb = GroupFeedback(
            coll_dim=coll_hidden,
            K=coll_k,
            output_dim=group_out_dim,
        ).to(device)

        self.set_phase2_modules(
            collusion_graph=collusion_mod,
            group_feedback=group_fb,
            twitter_id_to_global_idx=twitter_id_to_global_idx or {},
        )


    def forward_phase1(
        self,
        batch_data,
        time_mappings: Optional[List] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward_phase1."""

        h_prop, stage_info = self.prop_encoder.forward_batch(batch_data, time_mappings)
        # h_prop: [B, prop_dim]


        node_depth = getattr(batch_data, 'node_depth', None)
        z_u, r_u, p_user = self.user_risk_encoder(
            batch_data.x, batch_data.edge_index, batch_data.batch,
            node_depth=node_depth,
        )
        # z_u: [N, user_dim], r_u: [N], p_user: [B]


        z_u_pooled = risk_weighted_global_pool(
            z_u, r_u, batch_data.batch, batch_data.num_graphs
        )   # [B, user_dim]

        p_hat_0 = self.init_classifier(h_prop, z_u_pooled)   # [B]

        return h_prop, z_u, r_u, p_user, p_hat_0


    def forward(
        self,
        batch_data,
        time_mappings: Optional[List] = None,
        collusion_data: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward."""

        h_prop, z_u, r_u, p_user, p_hat_0 = self.forward_phase1(
            batch_data, time_mappings
        )
        # h_prop: [B, prop_dim], z_u: [N, user_dim]


        h_prop_mod, _ = self.risk_modulated_conv(
            batch_data.x,
            batch_data.edge_index,
            r_u,
            batch_data.batch,
            batch_data.num_graphs,
        )


        device = h_prop.device


        h_group = torch.zeros(
            batch_data.num_graphs, self.group_dim, device=device
        )

        if (
            self._use_phase2
            and collusion_data is not None
            and self.collusion_graph is not None
        ):
            coll_ei = collusion_data['edge_index'].to(device)
            coll_ew = collusion_data['edge_weight'].to(device)

            global_z = collusion_data.get('global_z_u', None)

            if global_z is not None:
                global_z = global_z.to(device)


                M_uv = self.collusion_graph.sparse_gate(global_z, coll_ei, coll_ew)
                eff_w = M_uv * coll_ew


                h_out_nodes = self.collusion_graph.gnn(global_z, coll_ei, eff_w)


                S   = self.collusion_graph.community_assign(h_out_nodes)
                H_C = self.collusion_graph.community_assign.get_cluster_features(
                    h_out_nodes, S
                )


                c_u_scores = self.collusion_graph.conf_scorer(h_out_nodes)


                # ── GroupFeedback ──────────────────────────────────────────
                if self._twitter_id_to_global_idx is not None:
                    h_group, _ = self.group_feedback(
                        H_C,
                        c_u_scores,
                        S,
                        batch_data,
                        h_out_nodes,               # per-user node embeddings
                        self._twitter_id_to_global_idx,
                    )


        z_u_pooled = risk_weighted_global_pool(
            z_u, r_u, batch_data.batch, batch_data.num_graphs
        )   # [B, user_dim]


        p_hat = self.final_classifier(h_prop_mod, z_u_pooled, h_group)

        return p_hat, p_user, p_hat_0


    @torch.no_grad()
    def predict(
        self,
        batch_data,
        time_mappings: Optional[List] = None,
        collusion_data: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run predict."""
        p_hat, _, _ = self.forward(batch_data, time_mappings, collusion_data)
        predictions = (p_hat >= 0.5).long()
        return predictions, p_hat


    def get_node_features(
        self, data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run get_node_features."""
        with torch.no_grad():
            prop_node_feat = self.prop_encoder.get_node_embeddings(
                data.x, data.edge_index
            )
            trust_node_feat = self.user_risk_encoder.encoder(
                data.x, data.edge_index, data.batch,
                return_node_embeddings=True,
            )
        return prop_node_feat, trust_node_feat


    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def phase1_parameters(self):
        """Run phase1_parameters."""
        modules = [
            self.prop_encoder,
            self.user_risk_encoder,
            self.risk_modulated_conv,
            self.init_classifier,
            self.final_classifier,
        ]
        for m in modules:
            yield from m.parameters()

    def phase2_parameters(self):
        """Run phase2_parameters."""
        modules = []
        if self.collusion_graph is not None:
            modules.append(self.collusion_graph)
        if self.group_feedback is not None:
            modules.append(self.group_feedback)
        for m in modules:
            yield from m.parameters()


class ImprovedIntegratedModel(MRDETGMModel):
    """ImprovedIntegratedModel module."""
    def forward(self, data, time_mappings=None):
        """Run forward."""
        p_hat, p_user, p_hat_0 = super().forward(data, time_mappings)


        logits = torch.stack([1 - p_hat, p_hat], dim=-1)
        return logits, p_hat_0, p_user   # prop_feat ≈ p̂_0, trust_feat ≈ p_user
