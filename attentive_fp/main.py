from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from attentive_fp.model import AttentiveFPModel


MODEL_SPEC = ModelSpec(
    name="AttentiveFP",
    slug="attentive_fp",
    uses_dual_features=False,
    builder=lambda: AttentiveFPModel(hidden_dim=200, num_layers=3, num_timesteps=3, dropout=0.5),
    default_hparams={
        "batch_size": 128,
        "lr": 10 ** (-3.5),
        "weight_decay": 1e-3,
        "num_epochs": 100,
        "patience": 15,
    },
    notes="Uses the official PyG AttentiveFP implementation.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
