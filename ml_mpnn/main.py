from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from ml_mpnn.model import MLMPNNModel


MODEL_SPEC = ModelSpec(
    name="ML-MPNN",
    slug="ml_mpnn",
    uses_dual_features=False,
    builder=lambda: MLMPNNModel(hidden_dim=128, num_layers=3, dropout=0.0),
    default_hparams={
        "batch_size": 64,
        "lr": 5e-4,
        "weight_decay": 5e-4,
        "num_epochs": 100,
        "patience": 15,
    },
    notes="Paper-guided multi-level message passing implementation from AdvProp.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
