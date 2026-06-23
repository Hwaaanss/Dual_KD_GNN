from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise SystemExit("Optuna is required for this script. Install it with: pip install optuna") from exc

from common.config import DEFAULT_SEED, get_device, set_seed
from common.data import create_datasets
from common.datasets import (
    DEFAULT_DATASET,
    available_datasets,
    get_dataset_spec,
    resolve_target_columns,
)
from common.io_utils import ensure_dir, save_json, save_run_artifacts
from common.plotting import plot_training_curves
from common.trainer import Trainer
from dual_kd_gnn.model import DoubleGCNTransformerModel


MODEL_NAME = "Double_GCN_Transformer"
MODEL_SLUG = "dual_kd_gnn"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune dual_kd_gnn with Optuna and save reproducible results.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=available_datasets(),
        help="Registered dataset to tune on. Resolves data path, SMILES column, and target tasks.",
    )
    parser.add_argument("--data-path", default=None, help="Override the CSV path inferred from --dataset.")
    parser.add_argument("--dataset-name", default=None, help="Label used for saved run directories. Defaults to --dataset.")
    parser.add_argument("--smiles-column", default=None, help="Override the SMILES column inferred from --dataset.")
    parser.add_argument("--target-columns", nargs="+", default=None, help="Override the target columns inferred from --dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda", help="Compute device. Defaults to cuda; pass cpu/mps/cuda:N to override.")
    parser.add_argument("--study-name", default="dual_kd_gnn_optuna")
    parser.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to a local SQLite DB.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / MODEL_SLUG / "optuna")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Maximum tuning time in seconds.")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--pruner", choices=["hyperband", "median", "none"], default="hyperband")
    parser.add_argument("--pruner-warmup-steps", type=int, default=8)
    parser.add_argument("--gcn-pretrain-epochs", type=int, default=150)
    parser.add_argument("--transformer-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--replay-best", type=Path, default=None, help="Train once from a saved best_config.json.")
    parser.add_argument("--replay-run-name", default=None, help="Run directory name for --replay-best.")
    return parser


def build_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(seed=args.seed, multivariate=True)


def build_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_steps)
    return optuna.pruners.HyperbandPruner(min_resource=max(args.pruner_warmup_steps, 1), reduction_factor=3)


def storage_url_for(args: argparse.Namespace, study_dir: Path) -> str:
    if args.storage:
        return args.storage
    db_path = study_dir / "study.db"
    return f"sqlite:///{db_path.resolve()}"


# ---------------------------------------------------------------------------
# Per-dataset Optuna search space.
#
# BASE_SEARCH_SPACE defines the default range for every tunable parameter.
# DATASET_SEARCH_SPACES holds per-dataset overrides: only the listed keys
# replace the base entry; everything else falls back to BASE_SEARCH_SPACE.
# resolve_search_space(dataset) merges the two, and run_tuning() selects the
# dataset automatically from --dataset, so no extra flag is needed.
#
# Entry formats consumed by suggest_param():
#   {"type": "categorical", "choices": [...]}
#   {"type": "int",   "low": int,   "high": int}
#   {"type": "float", "low": float, "high": float, "log": bool (default False)}
# ---------------------------------------------------------------------------

BASE_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # model kwargs
    "gnn_hidden": {"type": "categorical", "choices": [128, 192, 256, 384]},
    "dim_ff_multiplier": {"type": "categorical", "choices": [2, 3, 4]},
    "gnn_layers": {"type": "int", "low": 2, "high": 4},
    "gnn_dropout": {"type": "float", "low": 0.1, "high": 0.55},
    "nhead": {"type": "categorical", "choices": [4, 8]},
    "tf_layers": {"type": "int", "low": 1, "high": 3},
    "tf_dropout": {"type": "float", "low": 0.05, "high": 0.45},
    "ih_rank": {"type": "categorical", "choices": [16, 32, 64]},
    "ih_symmetric": {"type": "categorical", "choices": [True, False]},
    "ih_proj_dim": {"type": "categorical", "choices": [0, 128, 256]},
    "ih_num_prototypes": {"type": "categorical", "choices": [3, 4, 6, 8, 12]},
    "ih_assignment_mode": {"type": "categorical", "choices": ["hard", "soft", "sparse"]},
    "ih_diversity_weight": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "info_nce_temperature": {"type": "categorical", "choices": [0.1, 0.2, 0.5]},
    # hparams
    "batch_size": {"type": "categorical", "choices": [64, 128]},
    "transformer_lr": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 3e-3, "log": True},
    "pretrain_lr": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
    "ema_decay": {"type": "float", "low": 0.95, "high": 0.999},
    "ema_decay_init": {"type": "categorical", "choices": [0.90, 0.95, 0.98, 0.99]},
    "distill_weight": {"type": "float", "low": 1e-3, "high": 0.2, "log": True},
    "cross_distill_weight": {"type": "categorical", "choices": [0.02, 0.05, 0.1]},
}

DATASET_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    # bace: target test 0.900 (val ~= test, no overfit, transformer stage carries the
    # signal). Push capacity up and bias toward the regions the tuned run preferred.
    "bace": {
        "gnn_hidden": {"type": "categorical", "choices": [256, 384, 512]},
        "tf_layers": {"type": "int", "low": 2, "high": 4},
        "tf_dropout": {"type": "float", "low": 0.1, "high": 0.3},
        "ih_proj_dim": {"type": "categorical", "choices": [0, 128]},
        "ih_num_prototypes": {"type": "categorical", "choices": [4, 6, 8]},
        "info_nce_temperature": {"type": "categorical", "choices": [0.2, 0.5, 1.0]},
        "transformer_lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
        "pretrain_lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
    },
    # tox21: target test 0.860 (largest gap, 12 tasks, mild underfit). Add capacity and
    # more codebook prototypes; keep dropout in the higher band the tuned run liked.
    "tox21": {
        "gnn_hidden": {"type": "categorical", "choices": [128, 192, 256, 384, 512]},
        "tf_layers": {"type": "int", "low": 2, "high": 4},
        "tf_dropout": {"type": "float", "low": 0.2, "high": 0.5},
        "ih_proj_dim": {"type": "categorical", "choices": [128, 256]},
        "ih_num_prototypes": {"type": "categorical", "choices": [8, 12, 16, 24]},
        "ema_decay": {"type": "float", "low": 0.97, "high": 0.999},
        "info_nce_temperature": {"type": "categorical", "choices": [0.2, 0.5, 1.0]},
        "distill_weight": {"type": "float", "low": 5e-4, "high": 0.05, "log": True},
        "transformer_lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
    },
    # clintox: kept on BASE_SEARCH_SPACE unchanged (do not adjust).
    "clintox": {},
    # bbbp / sider: not yet specialized; inherit the base space.
    "bbbp": {},
    "sider": {},
}


def resolve_search_space(dataset: str) -> dict[str, dict[str, Any]]:
    """Merge BASE_SEARCH_SPACE with the dataset-specific overrides (if any)."""
    space = {name: dict(spec) for name, spec in BASE_SEARCH_SPACE.items()}
    for name, spec in DATASET_SEARCH_SPACES.get(dataset, {}).items():
        space[name] = dict(spec)
    return space


def suggest_param(trial: optuna.Trial, name: str, space: dict[str, dict[str, Any]]) -> Any:
    spec = space[name]
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    raise ValueError(f"Unknown search-space type for {name!r}: {kind!r}")


def sample_model_kwargs(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gnn_hidden = suggest_param(trial, "gnn_hidden", space)
    ff_multiplier = suggest_param(trial, "dim_ff_multiplier", space)
    return {
        "gnn_hidden": gnn_hidden,
        "gnn_layers": suggest_param(trial, "gnn_layers", space),
        "gnn_dropout": suggest_param(trial, "gnn_dropout", space),
        "nhead": suggest_param(trial, "nhead", space),
        "tf_layers": suggest_param(trial, "tf_layers", space),
        "dim_ff": gnn_hidden * ff_multiplier,
        "tf_dropout": suggest_param(trial, "tf_dropout", space),
        "ih_rank": suggest_param(trial, "ih_rank", space),
        "ih_symmetric": suggest_param(trial, "ih_symmetric", space),
        "ih_proj_dim": suggest_param(trial, "ih_proj_dim", space),
        "ih_num_prototypes": suggest_param(trial, "ih_num_prototypes", space),
        "ih_assignment_mode": suggest_param(trial, "ih_assignment_mode", space),
        "ih_diversity_weight": suggest_param(trial, "ih_diversity_weight", space),
        "info_nce_temperature": suggest_param(trial, "info_nce_temperature", space),
    }


def sample_hparams(
    trial: optuna.Trial, space: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    transformer_lr = suggest_param(trial, "transformer_lr", space)
    return {
        "batch_size": suggest_param(trial, "batch_size", space),
        "lr": transformer_lr,
        "weight_decay": suggest_param(trial, "weight_decay", space),
        "num_epochs": max(args.gcn_pretrain_epochs, args.transformer_epochs),
        "patience": args.patience,
        "gcn_pretrain_epochs": args.gcn_pretrain_epochs,
        "transformer_epochs": args.transformer_epochs,
        "pretrain_lr": suggest_param(trial, "pretrain_lr", space),
        "transformer_lr": transformer_lr,
        "ema_decay": suggest_param(trial, "ema_decay", space),
        "ema_decay_init": suggest_param(trial, "ema_decay_init", space),
        "distill_weight": suggest_param(trial, "distill_weight", space),
        "cross_distill_weight": suggest_param(trial, "cross_distill_weight", space),
    }


def build_epoch_callback(trial: optuna.Trial):
    def callback(event: dict[str, object]) -> None:
        if event.get("phase") != "stage2_transformer":
            return
        value = float(event["val_metric"])
        step = int(event["epoch"])
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at transformer epoch {step} with val AUC {value:.6f}")

    return callback


def build_metrics(
    *,
    dataset_name: str,
    target_columns: list[str],
    best_val_auc: float,
    best_epoch: int,
    model: DoubleGCNTransformerModel,
    elapsed_seconds: float,
    status: str,
    test_roc_auc: float | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "model_name": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "dataset_name": dataset_name,
        "target_columns": target_columns,
        "num_targets": len(target_columns),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_roc_auc": math.nan if test_roc_auc is None else test_roc_auc,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "uses_dual_features": True,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": status,
    }
    return metrics


def best_seen_by_trainer(trainer: Trainer) -> tuple[float, int]:
    if trainer.val_aucs:
        best_val_auc = max(trainer.val_aucs)
        best_epoch = trainer.val_aucs.index(best_val_auc) + 1
        return float(best_val_auc), int(best_epoch)
    return float(trainer.best_val_auc), int(trainer.best_epoch)


def train_once(
    *,
    train_dataset,
    val_dataset,
    test_dataset,
    target_columns: list[str],
    dataset_name: str,
    seed: int,
    device_name: str | None,
    model_kwargs: dict[str, Any],
    hparams: dict[str, Any],
    run_dir: Path,
    metadata: dict[str, object],
    epoch_callback=None,
    evaluate_test: bool = False,
    save_weights: bool = False,
) -> dict[str, object]:
    set_seed(seed)
    device = get_device(device_name)
    model = DoubleGCNTransformerModel(num_classes=len(target_columns), **model_kwargs)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_classes=len(target_columns),
        epoch_callback=epoch_callback,
        **hparams,
    )

    start = time.time()
    status = "complete"
    try:
        trainer.train()
    except optuna.TrialPruned:
        status = "pruned"
        elapsed = time.time() - start
        best_val_auc, best_epoch = best_seen_by_trainer(trainer)
        metrics = build_metrics(
            dataset_name=dataset_name,
            target_columns=target_columns,
            best_val_auc=best_val_auc,
            best_epoch=best_epoch,
            model=model,
            elapsed_seconds=elapsed,
            status=status,
        )
        save_run_artifacts(run_dir, trainer.build_history_rows(), metrics, metadata={**metadata, "status": status})
        raise

    test_auc = None
    if evaluate_test:
        test_auc = float(trainer.evaluate(test_dataset, batch_size=int(hparams["batch_size"])))

    elapsed = time.time() - start
    metrics = build_metrics(
        dataset_name=dataset_name,
        target_columns=target_columns,
        best_val_auc=float(trainer.best_val_auc),
        best_epoch=int(trainer.best_epoch),
        model=model,
        elapsed_seconds=elapsed,
        status=status,
        test_roc_auc=test_auc,
    )
    history_rows = trainer.build_history_rows()
    save_run_artifacts(run_dir, history_rows, metrics, metadata={**metadata, "status": status})
    if save_weights:
        import torch

        torch.save(trainer.best_state, run_dir / "model_weights.pt")
        plot_training_curves(history_rows, run_dir / "training_curves.png", title=f"{MODEL_SLUG} | {dataset_name}")
    return metrics


def save_trials_csv(study: optuna.Study, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "number",
                "state",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration_seconds",
                "params_json",
                "user_attrs_json",
            ],
        )
        writer.writeheader()
        for trial in study.trials:
            duration = trial.duration.total_seconds() if trial.duration is not None else math.nan
            writer.writerow(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value if trial.value is not None else math.nan,
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else "",
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else "",
                    "duration_seconds": duration,
                    "params_json": json.dumps(trial.params, ensure_ascii=True, sort_keys=True),
                    "user_attrs_json": json.dumps(trial.user_attrs, ensure_ascii=True, sort_keys=True),
                }
            )


def save_best_config(study: optuna.Study, study_dir: Path, base_config: dict[str, object]) -> Path | None:
    save_trials_csv(study, study_dir / "trials.csv")
    try:
        best_trial = study.best_trial
    except ValueError:
        return None

    config = {
        **base_config,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "model_kwargs": best_trial.user_attrs["model_kwargs"],
        "hparams": best_trial.user_attrs["hparams"],
        "metric": "best_val_auc",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    best_config_path = study_dir / "best_config.json"
    config["replay_command"] = f"python dual_kd_gnn/tune_optuna.py --replay-best {best_config_path}"
    save_json(best_config_path, config)
    save_json(study_dir / "best_trial.json", {"number": best_trial.number, "value": best_trial.value})
    return best_config_path


def run_tuning(args: argparse.Namespace) -> None:
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

    study_dir = ensure_dir(args.output_dir / args.study_name)
    storage_url = storage_url_for(args, study_dir)
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=args.seed,
        dual=True,
        smiles_column=smiles_column,
    )

    sampler = build_sampler(args)
    pruner = build_pruner(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Auto-select the search space from the chosen dataset (falls back to BASE).
    search_space = resolve_search_space(spec.name)
    specialized = spec.name in DATASET_SEARCH_SPACES and DATASET_SEARCH_SPACES[spec.name]
    print(
        f"Search space for '{spec.name}': "
        f"{'dataset-specific overrides applied' if specialized else 'base space (no overrides)'}"
    )

    base_config: dict[str, object] = {
        "study_name": args.study_name,
        "storage": storage_url,
        "dataset": spec.name,
        "data_path": str(Path(data_path).resolve()),
        "dataset_name": dataset_name,
        "smiles_column": smiles_column,
        "target_columns": target_columns,
        "seed": args.seed,
        "device": args.device,
        "search_space": search_space,
    }
    save_json(study_dir / "study_config.json", base_config)

    def objective(trial: optuna.Trial) -> float:
        model_kwargs = sample_model_kwargs(trial, search_space)
        hparams = sample_hparams(trial, search_space, args)
        trial.set_user_attr("model_kwargs", model_kwargs)
        trial.set_user_attr("hparams", hparams)

        trial_dataset_name = f"{dataset_name}_optuna_trial_{trial.number:04d}"
        trial_dir = ensure_dir(study_dir / "trials" / f"trial_{trial.number:04d}")
        save_json(
            trial_dir / "trial_config.json",
            {
                "trial_number": trial.number,
                "dataset_name": trial_dataset_name,
                "model_kwargs": model_kwargs,
                "hparams": hparams,
                "seed": args.seed,
            },
        )

        metadata = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "study_name": args.study_name,
            "trial_number": trial.number,
            "model_kwargs": model_kwargs,
            "hparams": hparams,
            "device": str(get_device(args.device)),
        }
        metrics = train_once(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            target_columns=target_columns,
            dataset_name=trial_dataset_name,
            seed=args.seed,
            device_name=args.device,
            model_kwargs=model_kwargs,
            hparams=hparams,
            run_dir=trial_dir,
            metadata=metadata,
            epoch_callback=build_epoch_callback(trial),
            evaluate_test=False,
        )
        return float(metrics["best_val_auc"])

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    best_config_path = save_best_config(study, study_dir, base_config)
    if best_config_path is None:
        print(f"No completed trials yet. Study artifacts are in: {study_dir}")
    else:
        print(f"Saved Optuna study artifacts to: {study_dir}")
        print(f"Saved replayable best config to: {best_config_path}")


def run_replay(args: argparse.Namespace) -> None:
    config = json.loads(args.replay_best.read_text(encoding="utf-8"))
    data_path = args.data_path or config["data_path"]
    dataset_name = args.dataset_name or config["dataset_name"]
    target_columns = list(config["target_columns"])
    smiles_column = args.smiles_column or config.get("smiles_column", "smiles")
    seed = int(config["seed"])
    device_name = args.device if args.device is not None else config.get("device")
    model_kwargs = dict(config["model_kwargs"])
    hparams = dict(config["hparams"])

    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=seed,
        dual=True,
        smiles_column=smiles_column,
    )
    run_name = args.replay_run_name or f"{dataset_name}_optuna_best"
    run_dir = PROJECT_ROOT / MODEL_SLUG / "runs" / run_name
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "replay_best_config": str(args.replay_best.resolve()),
        "data_path": str(Path(data_path).resolve()),
        "smiles_column": smiles_column,
        "model_kwargs": model_kwargs,
        "hparams": hparams,
        "device": str(get_device(device_name)),
    }
    metrics = train_once(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        target_columns=target_columns,
        dataset_name=run_name,
        seed=seed,
        device_name=device_name,
        model_kwargs=model_kwargs,
        hparams=hparams,
        run_dir=run_dir,
        metadata=metadata,
        evaluate_test=True,
        save_weights=True,
    )
    print(f"Saved replay run artifacts to: {run_dir}")
    print(f"  Best Val AUC: {float(metrics['best_val_auc']):.4f}")
    print(f"  Test ROC-AUC: {float(metrics['test_roc_auc']):.4f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.replay_best is not None:
        run_replay(args)
    else:
        run_tuning(args)


if __name__ == "__main__":
    main()
