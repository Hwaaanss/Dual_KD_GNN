from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class FPGNNModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 127,
        fp_dim: int = 1489,
        hidden_dim: int = 200,
        num_layers: int = 3,
        num_heads: int = 4,
        gnn_dropout: float = 0.5,
        fpn_hidden_dim: int = 256,
        fpn_dropout: float = 0.3,
        graph_ratio: float = 0.5,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = node_dim if layer_idx == 0 else hidden_dim
            self.gnn_layers.append(
                GATConv(in_dim, head_dim, heads=num_heads, concat=True, dropout=gnn_dropout)
            )
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        self.fp_net = nn.Sequential(
            nn.Linear(fp_dim, fpn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fpn_dropout),
            nn.Linear(fpn_hidden_dim, hidden_dim),
        )
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.graph_ratio = graph_ratio
        self.gnn_dropout = gnn_dropout
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(max(gnn_dropout, fpn_dropout)),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        h = data.x
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h = norm(F.elu(conv(h, data.edge_index)))
            h = F.dropout(h, p=self.gnn_dropout, training=self.training)

        graph_emb = global_mean_pool(self.graph_proj(h), data.batch)
        fp_emb = self.fp_net(data.fp_fpgnn)
        fused = torch.cat([self.graph_ratio * graph_emb, (1.0 - self.graph_ratio) * fp_emb], dim=-1)
        return self.fuse(fused)
