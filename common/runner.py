from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch

from common.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_SEED,
    DEFAULT_TARGET_COLUMNS,
    NUM_CLASSES,
    ModelSpec,
    get_device,
    set_seed,
)
from common.data import create_datasets
from common.io_utils import save_run_artifacts
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
    "distill_weight",
]


def add_shared_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-path", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Label used for saved run directories.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--device", default=None, help="Explicit device, e.g. cpu, mps, or cuda:0.")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=DEFAULT_TARGET_COLUMNS,
        help="Target columns to train on.",
    )
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
    parser.add_argument("--distill-weight", type=float, default=None)
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
        "distill_weight": args.distill_weight,
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
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(device_name)
    hparams = merge_hparams(spec.default_hparams, overrides)
    model_kwargs = model_kwargs or {}

    print(f"Using device: {device}")
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=seed,
        dual=spec.uses_dual_features,
    )

    model = spec.builder(**model_kwargs)
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
            "hparams": hparams,
            "model_kwargs": model_kwargs,
            "device": str(device),
            "model_notes": spec.notes,
        },
    )

    print(f"Saved run artifacts to: {run_dir}")
    print(f"  Best Val AUC: {trainer.best_val_auc:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")
    return metrics


def run_from_cli(spec: ModelSpec, model_dir: Path) -> dict[str, Any]:
    parser = build_single_model_parser(spec)
    args = parser.parse_args()
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
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        seed=args.seed,
        device_name=args.device,
        target_columns=args.target_columns,
        model_dir=model_dir,
        overrides=hparam_overrides,
        model_kwargs=model_kwargs,
    )
