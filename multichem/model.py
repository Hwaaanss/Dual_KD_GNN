from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

from common.graph_utils import build_line_graph, scatter_sum, to_dense_batch_manual


class MultiChemAtomLayer(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, node_h: torch.Tensor, edge_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return node_h
        src, dst = edge_index
        msg_input = torch.cat([node_h[src], edge_h], dim=-1)
        msg = F.elu(self.msg_proj(msg_input))
        score = self.attn(torch.cat([node_h[dst], msg], dim=-1))
        alpha = softmax(score, dst, num_nodes=node_h.size(0))
        context = scatter_sum(alpha * msg, dst, dim_size=node_h.size(0))
        new_node = self.gru(context, node_h)
        return self.ln(F.dropout(new_node, p=self.dropout, training=self.training))


class MultiChemBondLayer(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        edge_h: torch.Tensor,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        line_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_h.size(0) == 0:
            return edge_h
        src, dst = edge_index
        base = F.elu(self.msg_proj(torch.cat([node_h[src], node_h[dst]], dim=-1)))
        if line_edge_index.size(1) == 0:
            new_edge = self.gru(base, edge_h)
            return self.ln(F.dropout(new_edge, p=self.dropout, training=self.training))
        le_src, le_dst = line_edge_index
        edge_msg = F.elu(self.msg_proj(torch.cat([edge_h[le_src], node_h[src[le_src]]], dim=-1)))
        score = self.attn(torch.cat([edge_h[le_dst], edge_msg], dim=-1))
        alpha = softmax(score, le_dst, num_nodes=edge_h.size(0))
        context = scatter_sum(alpha * edge_msg, le_dst, dim_size=edge_h.size(0))
        new_edge = self.gru(base + context, edge_h)
        return self.ln(F.dropout(new_edge, p=self.dropout, training=self.training))


class MultiChemModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 127,
        edge_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        global_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        self.atom_layers = nn.ModuleList([MultiChemAtomLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.bond_layers = nn.ModuleList([MultiChemBondLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.global_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=global_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        node_h = F.relu(self.node_in(data.x))
        edge_h = (
            F.relu(self.edge_in(data.edge_attr))
            if data.edge_attr.size(0) > 0
            else data.x.new_zeros((0, self.node_in.out_features))
        )
        line_edge_index = build_line_graph(data.edge_index)
        for atom_layer, bond_layer in zip(self.atom_layers, self.bond_layers):
            edge_h = bond_layer(edge_h, node_h, data.edge_index, line_edge_index)
            node_h = atom_layer(node_h, edge_h, data.edge_index)

        dense_local, mask = to_dense_batch_manual(node_h, data.batch)
        global_out, _ = self.global_attn(
            dense_local,
            dense_local,
            dense_local,
            key_padding_mask=~mask,
        )
        global_out = (dense_local + F.dropout(global_out, p=self.dropout, training=self.training)) * mask.unsqueeze(-1)
        node_h = global_out[mask]
        graph_h = global_mean_pool(node_h, data.batch)
        return self.classifier(graph_h)
