from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP


class AttentiveFPModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 127,
        edge_dim: int = 12,
        hidden_dim: int = 200,
        num_layers: int = 3,
        num_timesteps: int = 3,
        dropout: float = 0.5,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        self.model = AttentiveFP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=num_classes,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index, data.edge_attr, data.batch)
