from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from fp_gnn.model import FPGNNModel


MODEL_SPEC = ModelSpec(
    name="FP-GNN",
    slug="fp_gnn",
    uses_dual_features=False,
    builder=lambda: FPGNNModel(
        hidden_dim=200,
        num_layers=3,
        num_heads=4,
        gnn_dropout=0.5,
        fpn_hidden_dim=256,
        fpn_dropout=0.3,
        graph_ratio=0.5,
    ),
    default_hparams={
        "batch_size": 64,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "num_epochs": 100,
        "patience": 15,
    },
    notes="Structure-preserving local FP-GNN implementation with graph and fingerprint fusion branches.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
