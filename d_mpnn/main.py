from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from d_mpnn.model import DMPNNModel


MODEL_SPEC = ModelSpec(
    name="D-MPNN",
    slug="d_mpnn",
    uses_dual_features=False,
    builder=lambda: DMPNNModel(hidden_dim=300, num_layers=3, dropout=0.0),
    default_hparams={
        "batch_size": 50,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "num_epochs": 100,
        "patience": 15,
    },
    notes="Lightweight local D-MPNN implementation following Chemprop-style directed bond message passing.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
