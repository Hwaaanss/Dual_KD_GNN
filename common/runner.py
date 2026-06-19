from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from common.config import (
    DEFAULT_SEED,
    ModelSpec,
    get_device,
    set_seed,
)
from common.data import create_datasets
from common.datasets import (
    DEFAULT_DATASET,
    available_datasets,
    get_dataset_spec,
    resolve_target_columns,
)
from common.io_utils import save_run_artifacts
from common.plotting import plot_training_curves
from common.trainer import Trainer


OVERRIDABLE_HPARAMS = [
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
    "ema_decay_init",
    "distill_weight",
    "cross_distill_weight",
]


def add_dataset_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=available_datasets(),
        help="Registered dataset to train on. Resolves data path, SMILES column, and target tasks.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the CSV path inferred from --dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Label used for saved run directories. Defaults to --dataset.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Override the SMILES column inferred from --dataset.",
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="Override the target columns inferred from --dataset.",
    )
    return parser


def add_general_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Compute device. Defaults to cuda; pass cpu/mps/cuda:N to override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--gcn-pretrain-epochs", type=int, default=None)
    parser.add_argument("--transformer-epochs", type=int, default=None)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--transformer-lr", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-decay-init", type=float, default=None)
    parser.add_argument("--distill-weight", type=float, default=None)
    parser.add_argument("--cross-distill-weight", type=float, default=None)
    return parser


def add_shared_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_dataset_arguments(parser)
    add_general_training_arguments(parser)
    return parser


def build_single_model_parser(spec: ModelSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {spec.name}.")
    add_shared_training_arguments(parser)
    if spec.add_model_arguments is not None:
        spec.add_model_arguments(parser)
    return parser


def collect_override_hparams(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "gcn_pretrain_epochs": args.gcn_pretrain_epochs,
        "transformer_epochs": args.transformer_epochs,
        "pretrain_lr": args.pretrain_lr,
        "transformer_lr": args.transformer_lr,
        "ema_decay": args.ema_decay,
        "ema_decay_init": args.ema_decay_init,
        "distill_weight": args.distill_weight,
        "cross_distill_weight": args.cross_distill_weight,
    }


def merge_hparams(defaults: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    hparams = dict(defaults)
    for key, value in (overrides or {}).items():
        if value is not None and key in defaults:
            hparams[key] = value
    return hparams


def run_experiment(
    spec: ModelSpec,
    data_path: str,
    dataset_name: str,
    seed: int,
    device_name: str | None,
    target_columns: list[str],
    model_dir: Path,
    overrides: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    smiles_column: str = "smiles",
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(device_name)
    hparams = merge_hparams(spec.default_hparams, overrides)
    model_kwargs = model_kwargs or {}
    num_classes = len(target_columns)

    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name} | targets ({num_classes}): {', '.join(target_columns)}")
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=seed,
        dual=spec.uses_dual_features,
        smiles_column=smiles_column,
    )

    model = spec.builder(num_classes=num_classes, **model_kwargs)
    train_start = time.time()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_classes=len(target_columns),
        **hparams,
    )
    trainer.train()
    batch_size = int(hparams.get("batch_size", 64))
    test_auc = float(trainer.evaluate(test_dataset, batch_size=batch_size))
    elapsed_seconds = time.time() - train_start

    history_rows = trainer.build_history_rows()
    run_dir = model_dir / "runs" / dataset_name
    metrics = {
        "model_name": spec.name,
        "model_slug": spec.slug,
        "dataset_name": dataset_name,
        "target_columns": target_columns,
        "num_targets": len(target_columns),
        "best_val_auc": float(trainer.best_val_auc),
        "best_epoch": int(trainer.best_epoch),
        "test_roc_auc": test_auc,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "uses_dual_features": spec.uses_dual_features,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    save_run_artifacts(
        run_dir=run_dir,
        history_rows=history_rows,
        metrics=metrics,
        metadata={
            "data_path": str(Path(data_path).resolve()),
            "smiles_column": smiles_column,
            "hparams": hparams,
            "model_kwargs": model_kwargs,
            "device": str(device),
            "model_notes": spec.notes,
        },
    )
    torch.save(trainer.best_state, run_dir / "model_weights.pt")
    plot_training_curves(history_rows, run_dir / "training_curves.png", title=f"{spec.slug} | {dataset_name}")

    print(f"Saved run artifacts to: {run_dir}")
    print(f"  Best Val AUC: {trainer.best_val_auc:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")
    return metrics


def resolve_dataset_inputs(args: argparse.Namespace) -> tuple[str, str, list[str], str]:
    """Resolve (data_path, dataset_name, target_columns, smiles_column).

    Registered ``--dataset`` metadata provides defaults; explicit CLI flags
    (``--data-path``, ``--target-columns``, ``--smiles-column``,
    ``--dataset-name``) override them.
    """
    spec = get_dataset_spec(args.dataset)
    data_path = args.data_path or str(spec.data_path())
    if not Path(data_path).exists():
        raise SystemExit(
            f"Dataset file not found: {data_path}\n"
            f"Download it first, e.g.: python scripts/download_data.py {spec.name}"
        )
    smiles_column = args.smiles_column or spec.smiles_column
    target_columns = (
        list(args.target_columns)
        if args.target_columns is not None
        else resolve_target_columns(spec, data_path)
    )
    dataset_name = args.dataset_name or spec.name
    return data_path, dataset_name, target_columns, smiles_column


def run_from_cli(spec: ModelSpec, model_dir: Path) -> dict[str, Any]:
    parser = build_single_model_parser(spec)
    args = parser.parse_args()
    data_path, dataset_name, target_columns, smiles_column = resolve_dataset_inputs(args)
    model_kwargs = spec.collect_model_kwargs(args) if spec.collect_model_kwargs is not None else {}
    hparam_overrides = spec.collect_hparam_overrides(args) if spec.collect_hparam_overrides is not None else {}
    hparam_overrides.update(
        {
            key: value
            for key, value in collect_override_hparams(args).items()
            if value is not None
        }
    )
    return run_experiment(
        spec=spec,
        data_path=data_path,
        dataset_name=dataset_name,
        seed=args.seed,
        device_name=args.device,
        target_columns=target_columns,
        model_dir=model_dir,
        overrides=hparam_overrides,
        model_kwargs=model_kwargs,
        smiles_column=smiles_column,
    )
