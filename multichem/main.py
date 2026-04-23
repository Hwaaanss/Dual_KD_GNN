from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from multichem.model import MultiChemModel


MODEL_SPEC = ModelSpec(
    name="MultiChem",
    slug="multichem",
    uses_dual_features=False,
    builder=lambda: MultiChemModel(hidden_dim=128, num_layers=3, global_heads=4, dropout=0.3),
    default_hparams={
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "num_epochs": 100,
        "patience": 15,
    },
    notes="Local modular MultiChem implementation guided by the official paper and repository layout.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
