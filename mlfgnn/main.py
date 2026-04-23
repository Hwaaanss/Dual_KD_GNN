from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from mlfgnn.model import MLFGNNModel


MODEL_SPEC = ModelSpec(
    name="MLFGNN",
    slug="mlfgnn",
    uses_dual_features=False,
    builder=lambda: MLFGNNModel(
        gat_hidden=110,
        transformer_layers=3,
        transformer_heads=19,
        transformer_head_dim=96,
        gnn_dropout=0.5,
        transformer_dropout=0.5,
        ff_dropout=0.05,
    ),
    default_hparams={
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 100,
        "patience": 15,
    },
    notes="Paper-guided multilevel fusion GNN implementation.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
