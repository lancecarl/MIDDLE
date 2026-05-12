"""Propagation encoders used by MR-DE-TGM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class PropagationEncoder(nn.Module):
    """PropagationEncoder module."""

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        """Run __init__."""
        super().__init__()


        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        """Run forward."""

        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)


        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)


        graph_emb = global_mean_pool(h, batch)

        return graph_emb

    def get_node_embeddings(self, x, edge_index):
        """Run get_node_embeddings."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)

        return h


class TemporalPropagationEncoder(nn.Module):
    """TemporalPropagationEncoder module."""

    def __init__(self, input_dim=768, hidden_dim=256,
                 output_dim=128, num_layers=2):
        super().__init__()

        self.gcn = GCNConv(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=output_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )

    def forward(self, temporal_graphs):
        """Run forward."""
        temporal_embs = []

        for g in temporal_graphs:
            h = self.gcn(g.x, g.edge_index)
            h = F.relu(h)
            graph_h = global_mean_pool(h, g.batch)
            temporal_embs.append(graph_h)


        seq = torch.stack(temporal_embs, dim=0)
        seq = seq.transpose(0, 1)  # [batch_size, T, hidden_dim]


        output, (h_n, c_n) = self.lstm(seq)


        return h_n[-1]  # [batch_size, output_dim]

    def get_node_embeddings(self, x, edge_index):
        """Run get_node_embeddings."""
        h = self.gcn(x, edge_index)
        h = F.relu(h)
        return h
