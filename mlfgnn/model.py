from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax

from common.graph_utils import build_dense_adjacency, scatter_sum, to_dense_batch_manual


class DyT(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 19,
        head_dim: int = 96,
        attn_dropout: float = 0.5,
        ff_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(d_model, self.inner_dim)
        self.k_proj = nn.Linear(d_model, self.inner_dim)
        self.v_proj = nn.Linear(d_model, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dyt1 = DyT(d_model)
        self.dyt2 = DyT(d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.lambda_a = nn.Parameter(torch.tensor(1.0))
        self.lambda_b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pad_mask = (~mask).unsqueeze(1).unsqueeze(2)
        score = score.masked_fill(pad_mask, float("-inf"))
        attn = F.softmax(score, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        mixed_attn = self.lambda_a * attn + self.lambda_b * adj_expanded
        mixed_attn = mixed_attn.masked_fill(pad_mask, 0.0)
        out = torch.matmul(self.attn_dropout(mixed_attn), v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.inner_dim)
        x = self.dyt1(x + self.out_proj(out))
        x = self.dyt2(x + self.ffn(x))
        return x * mask.unsqueeze(-1)


class MLFGNNLocalLayer(nn.Module):
    def __init__(self, hidden_dim: int = 110, edge_dim: int = 13, dropout: float = 0.5) -> None:
        super().__init__()
        self.neighbor_proj = nn.Linear(hidden_dim + edge_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return h
        src, dst = edge_index
        neighbor = F.leaky_relu(self.neighbor_proj(torch.cat([h[src], edge_attr], dim=-1)), 0.2)
        score = self.attn(torch.cat([h[dst], neighbor], dim=-1))
        alpha = softmax(score, dst, num_nodes=h.size(0))
        context = scatter_sum(alpha * neighbor, dst, dim_size=h.size(0))
        new_h = self.gru(F.elu(context), h)
        return self.ln(F.dropout(new_h, p=self.dropout, training=self.training))


class VirtualSuperNodeReadout(nn.Module):
    def __init__(self, hidden_dim: int = 110) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        graph_state = global_add_pool(h, batch)
        num_graphs = graph_state.size(0)
        updated = []
        for graph_idx in range(num_graphs):
            nodes = h[batch == graph_idx]
            if nodes.size(0) == 0:
                updated.append(graph_state[graph_idx])
                continue
            query = graph_state[graph_idx].unsqueeze(0).repeat(nodes.size(0), 1)
            score = self.attn(torch.cat([query, nodes], dim=-1))
            alpha = F.softmax(score, dim=0)
            context = (alpha * self.value_proj(nodes)).sum(dim=0)
            updated.append(self.gru(context, graph_state[graph_idx]))
        return torch.stack(updated, dim=0)


class MLFGNNModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 54,
        edge_dim: int = 13,
        fp_dim: int = 2346,
        gat_hidden: int = 110,
        transformer_layers: int = 3,
        transformer_heads: int = 19,
        transformer_head_dim: int = 96,
        gnn_dropout: float = 0.5,
        transformer_dropout: float = 0.5,
        ff_dropout: float = 0.05,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        self.gat_input = nn.Linear(node_dim, gat_hidden)
        self.gat_layers = nn.ModuleList(
            [MLFGNNLocalLayer(gat_hidden, edge_dim, gnn_dropout) for _ in range(transformer_layers)]
        )

        transformer_dim = transformer_heads * transformer_head_dim
        self.tf_input = nn.Linear(node_dim, transformer_dim)
        self.tf_layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    transformer_dim,
                    transformer_heads,
                    transformer_head_dim,
                    transformer_dropout,
                    ff_dropout,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.tf_output = nn.Linear(transformer_dim, gat_hidden)

        self.mix_gate = nn.Parameter(torch.tensor(0.5))
        self.readout = VirtualSuperNodeReadout(gat_hidden)

        self.fp_net = nn.Sequential(
            nn.Linear(fp_dim, gat_hidden * 2),
            nn.ReLU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(gat_hidden * 2, gat_hidden),
        )
        self.cross_attn = nn.MultiheadAttention(
            gat_hidden,
            num_heads=1,
            batch_first=True,
            dropout=transformer_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gat_hidden * 4, gat_hidden * 2),
            nn.ReLU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(gat_hidden * 2, num_classes),
        )

    def forward(self, data):
        x = data.x_mlfgnn
        edge_index = data.edge_index
        edge_attr = data.edge_attr_mlfgnn
        batch = data.batch

        local_h = F.relu(self.gat_input(x))
        for layer in self.gat_layers:
            local_h = layer(local_h, edge_index, edge_attr)

        dense_x, mask = to_dense_batch_manual(x, batch)
        adj = build_dense_adjacency(batch, edge_index, dense_x.size(1))
        global_h = self.tf_input(dense_x)
        for layer in self.tf_layers:
            global_h = layer(global_h, adj, mask)
        global_h = self.tf_output(global_h)[mask]

        alpha = torch.clamp(self.mix_gate, 0.0, 1.0)
        mixed_h = alpha * local_h + (1.0 - alpha) * global_h
        graph_repr = self.readout(mixed_h, batch)

        fp_repr = self.fp_net(data.fp_mlfgnn)
        graph_token = graph_repr.unsqueeze(1)
        fp_token = fp_repr.unsqueeze(1)
        graph_to_fp, _ = self.cross_attn(graph_token, fp_token, fp_token)
        fp_to_graph, _ = self.cross_attn(fp_token, graph_token, graph_token)
        fused = torch.cat(
            [
                graph_repr,
                fp_repr,
                graph_to_fp.squeeze(1),
                fp_to_graph.squeeze(1),
            ],
            dim=-1,
        )
        return self.classifier(fused)
