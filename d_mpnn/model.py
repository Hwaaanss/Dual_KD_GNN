from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from common.graph_utils import build_reverse_edge_index, scatter_sum


class DMPNNModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 127,
        edge_dim: int = 12,
        hidden_dim: int = 300,
        num_layers: int = 3,
        dropout: float = 0.0,
        desc_dim: int = 200,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.init_edge = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.atom_proj = nn.Linear(node_dim + hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + desc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_edges = edge_index.size(1)
        if num_edges == 0:
            atom_hidden = F.relu(
                self.atom_proj(torch.cat([x, x.new_zeros((x.size(0), self.hidden_dim))], dim=-1))
            )
            graph_hidden = global_mean_pool(atom_hidden, batch)
            return self.classifier(torch.cat([graph_hidden, data.rdkit_desc], dim=-1))

        src, dst = edge_index
        reverse_index = build_reverse_edge_index(edge_index)
        h0 = F.relu(self.init_edge(torch.cat([x[src], edge_attr], dim=-1)))
        h = h0
        for _ in range(max(self.num_layers - 1, 1)):
            msg = h.new_zeros((num_edges, self.hidden_dim))
            for edge_idx in range(num_edges):
                incoming = torch.where(dst == src[edge_idx])[0]
                if incoming.numel() > 0:
                    if reverse_index[edge_idx] >= 0:
                        incoming = incoming[incoming != reverse_index[edge_idx]]
                    if incoming.numel() > 0:
                        msg[edge_idx] = h[incoming].sum(dim=0)
            h = F.relu(h0 + self.msg_proj(msg))
            h = F.dropout(h, p=self.dropout, training=self.training)

        atom_msg = scatter_sum(h, dst, dim_size=x.size(0))
        atom_hidden = F.relu(self.atom_proj(torch.cat([x, atom_msg], dim=-1)))
        atom_hidden = F.dropout(atom_hidden, p=self.dropout, training=self.training)
        graph_hidden = global_mean_pool(atom_hidden, batch)
        return self.classifier(torch.cat([graph_hidden, data.rdkit_desc], dim=-1))
