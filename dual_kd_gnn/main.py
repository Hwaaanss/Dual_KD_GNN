from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec
from common.runner import run_from_cli

from dual_kd_gnn.model import DoubleGCNTransformerModel


MODEL_KWARG_NAMES = [
    "gnn_hidden",
    "gnn_layers",
    "gnn_dropout",
    "nhead",
    "tf_layers",
    "dim_ff",
    "tf_dropout",
    "ih_rank",
    "ih_symmetric",
    "ih_proj_dim",
]
HPARAM_NAMES = [
    "batch_size",
    "lr",
    "weight_decay",
    "num_epochs",
    "patience",
    "gcn_pretrain_epochs",
    "transformer_epochs",
    "pretrain_lr",
    "transformer_lr",
    "ema_decay",
    "distill_weight",
]


def load_best_config(args) -> dict[str, object]:
    if args.best_config is None:
        return {}
    if not hasattr(args, "_best_config_data"):
        args._best_config_data = json.loads(args.best_config.read_text(encoding="utf-8"))
    return args._best_config_data


def add_dual_model_arguments(parser) -> None:
    parser.add_argument(
        "--best-config",
        type=Path,
        default=None,
        help="Path to an Optuna best_config.json containing model_kwargs and hparams.",
    )
    parser.add_argument("--gnn-hidden", type=int, default=None)
    parser.add_argument("--gnn-layers", type=int, default=None)
    parser.add_argument("--gnn-dropout", type=float, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--tf-layers", type=int, default=None)
    parser.add_argument("--dim-ff", type=int, default=None)
    parser.add_argument("--tf-dropout", type=float, default=None)
    parser.add_argument("--ih-rank", type=int, default=None)
    ih_symmetric = parser.add_mutually_exclusive_group()
    ih_symmetric.add_argument("--ih-symmetric", dest="ih_symmetric", action="store_true", default=None)
    ih_symmetric.add_argument("--ih-asymmetric", dest="ih_symmetric", action="store_false")
    parser.add_argument("--ih-proj-dim", type=int, default=None)


def collect_dual_model_kwargs(args) -> dict[str, object]:
    config = load_best_config(args)
    model_kwargs = {
        name: value
        for name, value in dict(config.get("model_kwargs", {})).items()
        if name in MODEL_KWARG_NAMES
    }
    model_kwargs.update({
        name: value
        for name in MODEL_KWARG_NAMES
        if (value := getattr(args, name)) is not None
    })
    return model_kwargs


def collect_dual_hparam_overrides(args) -> dict[str, object]:
    config = load_best_config(args)
    return {
        name: value
        for name, value in dict(config.get("hparams", {})).items()
        if name in HPARAM_NAMES
    }


MODEL_SPEC = ModelSpec(
    name="Double_GCN_Transformer",
    slug="dual_kd_gnn",
    uses_dual_features=True,
    builder=lambda **model_kwargs: DoubleGCNTransformerModel(**model_kwargs),
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
        "distill_weight": 0.05,
    },
    add_model_arguments=add_dual_model_arguments,
    collect_model_kwargs=collect_dual_model_kwargs,
    collect_hparam_overrides=collect_dual_hparam_overrides,
    notes="Dual-branch GCN + EMA teacher KD + transformer model with BCE task loss plus weighted KD during stage 1.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
