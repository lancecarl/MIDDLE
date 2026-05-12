"""Stage-aware temporal propagation encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import numpy as np
import math
from typing import Optional, List, Tuple, Dict


class PositionalEncoding(nn.Module):
    """PositionalEncoding module."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    """TemporalTransformerEncoder module."""

    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model, max_len=20)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self._transformer_dropout_p = dropout
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward."""
        x = self.pos_encoder(x)
        # return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return F.dropout(out, p=0.3, training=self.training)


class ImprovedTemporalPropagationEncoder(nn.Module):
    """ImprovedTemporalPropagationEncoder module."""


    STRUCT_FEAT_DIM = 7

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        gcn_dim: int = 64,
        transformer_hidden: int = 192,
        window_size: int = 3600,
    ):
        super().__init__()

        self.window_size = window_size
        self.gcn_dim = gcn_dim
        self.transformer_hidden = transformer_hidden


        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim // 4,
            heads=4,
            concat=True,
            dropout=0.3,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=gcn_dim,
            heads=1,
            concat=False,
            dropout=0.3,
        )
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(gcn_dim)
        self.dropout = nn.Dropout(0.3)


        self.struct_embed = nn.Sequential(
            nn.Linear(self.STRUCT_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.GELU(),
        )


        self.input_projection = nn.Linear(gcn_dim + 128, transformer_hidden)  # ★ 64→128

        self.temporal_transformer = TemporalTransformerEncoder(
            d_model=transformer_hidden,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.3,
        )
        # self.temporal_transformer = torch._dynamo.disable(self.temporal_transformer)


        half_dim = transformer_hidden // 2

        # s_t = σ( stage_proj( |h_t - h_{t-1}| ) )
        self.stage_proj = nn.Linear(transformer_hidden, 1)

        # α_t ∝ stage_attn_proj( tanh(W_h h_t + W_s s_t + W_a a_t) )
        self.W_h = nn.Linear(transformer_hidden, half_dim, bias=False)
        self.W_s = nn.Linear(1, half_dim, bias=False)
        self.W_a = nn.Linear(128, half_dim, bias=False)
        self.stage_attn_proj = nn.Linear(half_dim, 1)
        self.stage_attn_bias = nn.Parameter(torch.zeros(half_dim))


        self.attention_pooling = nn.Sequential(
            nn.Linear(transformer_hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.aggregation = 'stage_attention'


        self.output_proj = nn.Sequential(
            nn.Linear(transformer_hidden, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )


    def extract_substructures(
        self,
        data: Data,
        time_mapping: Optional[Dict],
    ) -> Tuple[List[Data], torch.Tensor]:
        """Run extract_substructures."""
        if time_mapping is None or len(time_mapping) == 0:
            dummy_feat = torch.zeros(1, self.STRUCT_FEAT_DIM).to(data.x.device)
            return [data], dummy_feat

        timestamps, node_ids = [], []
        for nid in range(data.x.shape[0]):
            if nid in time_mapping:
                ts = time_mapping[nid]
                if ts and ts != '':
                    try:
                        timestamps.append(int(ts))
                        node_ids.append(nid)
                    except Exception:
                        continue

        if len(timestamps) == 0:
            dummy_feat = torch.zeros(1, self.STRUCT_FEAT_DIM).to(data.x.device)
            return [data], dummy_feat

        min_time = min(timestamps)
        max_time = max(timestamps)
        num_windows = min(10, max(3, int((max_time - min_time) / self.window_size) + 1))
        window_edges = np.linspace(min_time, max_time, num_windows + 1)

        substructures: List[Data] = []
        struct_features: List[torch.Tensor] = []
        n_nodes_history: List[int] = []

        for i in range(num_windows):
            t_end = window_edges[i + 1]


            window_nodes = set()
            for nid, ts in zip(node_ids, timestamps):
                if ts <= t_end:
                    window_nodes.add(nid)

            if len(window_nodes) == 0:
                continue

            window_nodes.add(0)
            window_nodes = sorted(list(window_nodes))
            node_map = {old: new for new, old in enumerate(window_nodes)}

            edge_index = data.edge_index.cpu().numpy()
            sub_edges = [
                [node_map[src], node_map[dst]]
                for src, dst in edge_index.T
                if src in set(window_nodes) and dst in set(window_nodes)
            ]
            if len(sub_edges) == 0:
                continue

            sub_edges_t = torch.tensor(sub_edges, dtype=torch.long).t()
            sub_x = data.x[window_nodes]
            sub_data = Data(x=sub_x, edge_index=sub_edges_t)
            substructures.append(sub_data)

            prev_struct = substructures[-2] if len(substructures) > 1 else None


            feat = self._compute_structure_features(
                sub_data,
                prev_struct=prev_struct,
                prev_n_nodes_history=n_nodes_history.copy(),
            )
            struct_features.append(feat)
            n_nodes_history.append(len(window_nodes))

        if len(substructures) == 0:
            dummy_feat = torch.zeros(1, self.STRUCT_FEAT_DIM).to(data.x.device)
            return [data], dummy_feat

        struct_tensor = torch.stack(struct_features).to(data.x.device)
        return substructures, struct_tensor


    def _compute_structure_features(
        self,
        current_struct: Data,
        prev_struct: Optional[Data] = None,
        prev_n_nodes_history: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Run _compute_structure_features."""
        n_nodes = current_struct.x.shape[0]
        edge_index = current_struct.edge_index.cpu().numpy()
        levels = self._compute_levels(edge_index, n_nodes)

        max_level = max(levels.values()) if levels else 0
        level_counts: Dict[int, int] = {}
        for lv in levels.values():
            level_counts[lv] = level_counts.get(lv, 0) + 1

        depth = max_level + 1
        width = max(level_counts.values()) if level_counts else 1


        if prev_struct is not None:
            prev_n = prev_struct.x.shape[0]
            prev_edge = prev_struct.edge_index.cpu().numpy()
            prev_levels = self._compute_levels(prev_edge, prev_n)

            prev_max = max(prev_levels.values()) if prev_levels else 0
            prev_level_counts: Dict[int, int] = {}
            for lv in prev_levels.values():
                prev_level_counts[lv] = prev_level_counts.get(lv, 0) + 1

            prev_depth = prev_max + 1
            prev_width = max(prev_level_counts.values()) if prev_level_counts else 1

            delta_n = (n_nodes - prev_n) / max(prev_n, 1)
            delta_d = depth / max(prev_depth, 1) - 1.0
            delta_w = width / max(prev_width, 1) - 1.0
        else:
            delta_n = delta_d = delta_w = 0.0


        if prev_n_nodes_history and len(prev_n_nodes_history) > 0:
            mean_prev = np.mean(prev_n_nodes_history)
            burst = float(np.clip(n_nodes / (mean_prev + 1e-6), 0, 10) / 10.0)
        else:
            burst = 0.1


        n_norm   = float(np.log(1 + n_nodes) / 10.0)
        d_norm   = float(np.log(1 + depth) / 5.0)
        w_norm   = float(np.log(1 + width) / 5.0)

        return torch.tensor(
            [n_norm, d_norm, w_norm, delta_n, delta_d, delta_w, burst],
            dtype=torch.float32,
        )

    def _compute_levels(
        self,
        edge_index: np.ndarray,
        n_nodes: int,
    ) -> Dict[int, int]:
        """Run _compute_levels."""
        from collections import deque

        MAX_DEPTH = 30

        adj: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        if edge_index.shape[1] > 0:
            for i in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                adj[src].append(dst)

        levels = {0: 0}
        queue = deque([0])

        while queue:
            node = queue.popleft()
            current_level = levels[node]
            if current_level >= MAX_DEPTH:
                continue
            for neighbor in adj[node]:
                if neighbor not in levels:
                    levels[neighbor] = current_level + 1
                    queue.append(neighbor)

        return levels


    def aggregate_sequence(
        self,
        sequence: torch.Tensor,
        struct_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run aggregate_sequence."""
        T = sequence.shape[0]


        # s_t = σ( stage_proj( |h_t - h_{t-1}| ) )
        h_diff = torch.zeros_like(sequence)
        if T > 1:
            h_diff[1:] = torch.abs(sequence[1:] - sequence[:-1])

        s_t = torch.sigmoid(self.stage_proj(h_diff))   # [T, 1]


        if (
            self.aggregation == 'stage_attention'
            and struct_features is not None
            and struct_features.shape[0] == T
        ):

            struct_emb = self.struct_embed(struct_features)   # [T, 128]


            h_proj = self.W_h(sequence)                       # [T, D/2]
            s_proj = self.W_s(s_t)                            # [T, D/2]
            a_proj = self.W_a(struct_emb)                     # [T, D/2]

            combined = torch.tanh(
                h_proj + s_proj + a_proj + self.stage_attn_bias
            )                                                  # [T, D/2]
            scores  = self.stage_attn_proj(combined).squeeze(-1)  # [T]
            weights = F.softmax(scores, dim=0)                    # [T]

            aggregated = (sequence * weights.unsqueeze(-1)).sum(dim=0)  # [D]

        elif self.aggregation in ('mean',):
            aggregated = sequence.mean(dim=0)

        elif self.aggregation in ('max',):
            aggregated = sequence.max(dim=0)[0]

        elif self.aggregation in ('last',):
            aggregated = sequence[-1]

        else:

            weights_raw = self.attention_pooling(sequence)    # [T, 1]
            weights     = F.softmax(weights_raw, dim=0)
            aggregated  = (sequence * weights).sum(dim=0)

        return aggregated, s_t.squeeze(-1)


    def forward(
        self,
        data: Data,
        time_mapping: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward."""
        device = data.x.device


        substructures, struct_features = self.extract_substructures(data, time_mapping)

        if len(substructures) == 0:
            dummy_out = torch.zeros(self.output_proj[0].out_features, device=device)
            dummy_st  = torch.zeros(1, device=device)
            return dummy_out, dummy_st


        temporal_embeddings: List[torch.Tensor] = []
        for sub_data in substructures:
            sub_data = sub_data.to(device)

            h = self.gat1(sub_data.x, sub_data.edge_index)
            h = self.bn1(h)
            h = F.gelu(h)
            h = self.dropout(h)

            h = self.gat2(h, sub_data.edge_index)
            h = self.bn2(h)
            h = F.gelu(h)

            graph_emb = h.mean(dim=0)          # [gcn_dim]
            temporal_embeddings.append(graph_emb)


        struct_emb_seq = self.struct_embed(struct_features)  # [T, 128]


        temporal_stack = torch.stack(temporal_embeddings)    # [T, gcn_dim]
        combined = torch.cat([temporal_stack, struct_emb_seq], dim=1)  # [T, gcn+128]
        transformer_input = self.input_projection(combined).unsqueeze(0)  # [1, T, D_tr]


        transformer_output = self.temporal_transformer(transformer_input)  # [1, T, D_tr]
        transformer_output = transformer_output.squeeze(0)                  # [T, D_tr]


        aggregated, stage_factors = self.aggregate_sequence(
            transformer_output,
            struct_features=struct_features,
        )                                                     # [D_tr], [T]


        output = self.output_proj(aggregated.unsqueeze(0)).squeeze(0)  # [output_dim]

        return output, stage_factors


    def forward_batch(
        self,
        batch_data,
        time_mappings: Optional[List[Optional[Dict]]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run forward_batch."""
        batch_features: List[torch.Tensor] = []
        all_stage_factors: List[torch.Tensor] = []
        num_graphs = batch_data.num_graphs

        for i in range(num_graphs):

            mask = (batch_data.batch == i)
            node_indices = torch.where(mask)[0]

            edge_mask = torch.isin(batch_data.edge_index[0], node_indices)
            sub_edges = batch_data.edge_index[:, edge_mask]

            node_map = {old.item(): new for new, old in enumerate(node_indices)}
            if sub_edges.shape[1] > 0:
                sub_edges_ri = torch.tensor(
                    [[node_map[s.item()], node_map[d.item()]]
                     for s, d in sub_edges.t()],
                    dtype=torch.long,
                ).t().to(batch_data.x.device)
            else:
                sub_edges_ri = torch.zeros((2, 0), dtype=torch.long,
                                           device=batch_data.x.device)

            sub_x = batch_data.x[node_indices]
            sub_data = Data(x=sub_x, edge_index=sub_edges_ri)

            time_map = (
                time_mappings[i]
                if time_mappings and i < len(time_mappings)
                else None
            )

            feat, stage_f = self.forward(sub_data, time_map)

            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            batch_features.append(feat)
            all_stage_factors.append(stage_f)

        if not batch_features:
            out_dim = self.output_proj[0].out_features
            empty = torch.zeros(num_graphs, out_dim,
                                device=batch_data.x.device)
            return empty, {'stage_factors': [], 'max_stage_factor': torch.zeros(num_graphs),
                           'mean_stage_factor': torch.zeros(num_graphs)}

        batch_tensor = torch.cat(batch_features, dim=0)   # [B, output_dim]


        max_sf  = torch.stack([sf.max() if sf.numel() > 0 else sf.new_zeros(1).squeeze()
                               for sf in all_stage_factors])    # [B]
        mean_sf = torch.stack([sf.mean() if sf.numel() > 0 else sf.new_zeros(1).squeeze()
                               for sf in all_stage_factors])    # [B]

        stage_info = {
            'stage_factors':      all_stage_factors,
            'max_stage_factor':   max_sf,
            'mean_stage_factor':  mean_sf,
        }

        return batch_tensor, stage_info


    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run get_node_embeddings."""
        h = self.gat1(x, edge_index)
        h = self.bn1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.gelu(h)

        return h


class PropagationEncoder(nn.Module):
    """PropagationEncoder module."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2   = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.3)

        self._global_mean_pool = global_mean_pool

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return self._global_mean_pool(h, batch)

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h
