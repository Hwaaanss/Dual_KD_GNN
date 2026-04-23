from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from dual_kd_gnn.model import DoubleGCNTransformerModel


MODEL_SPEC = ModelSpec(
    name="Double_GCN_Transformer",
    slug="dual_kd_gnn",
    uses_dual_features=True,
    builder=lambda: DoubleGCNTransformerModel(),
    default_hparams={
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "num_epochs": 100,
        "patience": 15,
        "gcn_pretrain_epochs": 100,
        "transformer_epochs": 100,
        "pretrain_lr": 1e-3,
        "transformer_lr": 1e-3,
        "ema_decay": 0.99,
        "distill_weight": 1.0,
    },
    notes="Original experimental dual-branch GCN + EMA KD + transformer model from the notebook.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
