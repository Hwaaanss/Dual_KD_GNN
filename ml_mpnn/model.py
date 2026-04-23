from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from common.graph_utils import MLPBlock, scatter_mean, size_norm


class MLMPNNModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 127,
        edge_dim: int = 12,
        subgraph_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        desc_dim: int = 200,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        self.sub_in = nn.Linear(subgraph_dim, hidden_dim)

        self.edge_mlps = nn.ModuleList([MLPBlock(hidden_dim * 4, hidden_dim, dropout) for _ in range(num_layers)])
        self.edge2node_mlps = nn.ModuleList([MLPBlock(hidden_dim * 2, hidden_dim, dropout) for _ in range(num_layers)])
        self.node_mlps = nn.ModuleList([MLPBlock(hidden_dim * 3, hidden_dim, dropout) for _ in range(num_layers)])
        self.node2sub_mlps = nn.ModuleList([MLPBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.sub_msg_mlps = nn.ModuleList([MLPBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.sub_mlps = nn.ModuleList([MLPBlock(hidden_dim * 4, hidden_dim, dropout) for _ in range(num_layers)])
        self.graph_mlps = nn.ModuleList([MLPBlock(hidden_dim * 4, hidden_dim, dropout) for _ in range(num_layers)])

        self.edge_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.node_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.sub_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        sub_x, sub_edge_index = data.subgraph_x, data.subgraph_edge_index
        sub_batch, assign_index = data.subgraph_batch, data.assign_index

        node_h = F.relu(self.node_in(x))
        edge_h = F.relu(self.edge_in(edge_attr)) if edge_attr.size(0) > 0 else x.new_zeros((0, self.hidden_dim))
        sub_h = F.relu(self.sub_in(sub_x))

        if edge_h.size(0) > 0:
            edge_batch = batch[edge_index[0]]
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            graph_h = global_mean_pool(edge_h, edge_batch, size=num_graphs)
        else:
            edge_batch = torch.empty((0,), dtype=torch.long, device=batch.device)
            graph_h = global_mean_pool(node_h, batch)

        for layer in range(self.num_layers):
            if edge_h.size(0) > 0:
                src, dst = edge_index
                edge_input = torch.cat([edge_h, node_h[src], node_h[dst], graph_h[edge_batch]], dim=-1)
                edge_h_upd = self.edge_mlps[layer](edge_input)
                node_msg_input = torch.cat([edge_h_upd, node_h[src]], dim=-1)
                node_msg = scatter_mean(
                    self.edge2node_mlps[layer](node_msg_input),
                    dst,
                    dim_size=node_h.size(0),
                )
            else:
                edge_h_upd = edge_h
                node_msg = node_h.new_zeros(node_h.size())

            node_h_upd = self.node_mlps[layer](torch.cat([node_h, node_msg, graph_h[batch]], dim=-1))

            atom_idx, sub_idx = assign_index
            node_to_sub = scatter_mean(
                self.node2sub_mlps[layer](node_h_upd[atom_idx]),
                sub_idx,
                dim_size=sub_h.size(0),
            )
            if sub_edge_index.size(1) > 0:
                sub_src, sub_dst = sub_edge_index
                sub_neigh = scatter_mean(
                    self.sub_msg_mlps[layer](sub_h[sub_src]),
                    sub_dst,
                    dim_size=sub_h.size(0),
                )
            else:
                sub_neigh = sub_h.new_zeros(sub_h.size())

            sub_h_upd = self.sub_mlps[layer](
                torch.cat([sub_h, node_to_sub, sub_neigh, graph_h[sub_batch]], dim=-1)
            )

            edge_pool = (
                global_mean_pool(edge_h_upd, edge_batch, size=graph_h.size(0))
                if edge_h_upd.size(0) > 0
                else graph_h.new_zeros(graph_h.size())
            )
            node_pool = global_mean_pool(node_h_upd, batch)
            sub_pool = global_mean_pool(sub_h_upd, sub_batch)
            graph_h = self.graph_mlps[layer](torch.cat([graph_h, edge_pool, node_pool, sub_pool], dim=-1))

            if edge_h_upd.size(0) > 0:
                edge_h = self.edge_norms[layer](size_norm(edge_h_upd, edge_batch))
                edge_h = F.dropout(edge_h, p=self.dropout, training=self.training)
            else:
                edge_h = edge_h_upd
            node_h = self.node_norms[layer](size_norm(node_h_upd, batch))
            node_h = F.dropout(node_h, p=self.dropout, training=self.training)
            sub_h = self.sub_norms[layer](size_norm(sub_h_upd, sub_batch))
            sub_h = F.dropout(sub_h, p=self.dropout, training=self.training)

        return self.classifier(graph_h)
