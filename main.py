from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.config import get_project_root
from common.datasets import (
    DEFAULT_BENCHMARK_DATASETS,
    available_datasets,
    get_dataset_spec,
    resolve_target_columns,
)
from common.io_utils import ensure_dir
from common.runner import add_general_training_arguments, collect_override_hparams, run_experiment
from dual_kd_gnn.main import (
    MODEL_SPEC,
    add_dual_model_arguments,
    collect_dual_hparam_overrides,
    collect_dual_model_kwargs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the dual_kd_gnn model across multiple MoleculeNet datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_BENCHMARK_DATASETS,
        choices=available_datasets(),
        help="Datasets to benchmark sequentially. Defaults to BACE, BBBP, SIDER, Tox21, ClinTox.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=get_project_root() / "results" / "artifacts",
        help="Directory for the aggregated benchmark summary table.",
    )
    add_general_training_arguments(parser)
    add_dual_model_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    spec = MODEL_SPEC
    model_kwargs = collect_dual_model_kwargs(args)
    hparam_overrides = collect_dual_hparam_overrides(args)
    hparam_overrides.update(
        {key: value for key, value in collect_override_hparams(args).items() if value is not None}
    )

    model_dir = get_project_root() / spec.slug
    summary_rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for name in args.datasets:
        dataset_spec = get_dataset_spec(name)
        data_path = dataset_spec.data_path()
        if not data_path.exists():
            print(
                f"[skip] {name}: dataset file not found at {data_path}. "
                f"Run: python scripts/download_data.py {name}"
            )
            skipped.append(name)
            continue

        target_columns = resolve_target_columns(dataset_spec, data_path)
        print(f"\n{'=' * 72}")
        print(f"Benchmarking {spec.name} on {name} ({len(target_columns)} task(s))")
        print(f"{'=' * 72}")
        metrics = run_experiment(
            spec=spec,
            data_path=str(data_path),
            dataset_name=dataset_spec.name,
            seed=args.seed,
            device_name=args.device,
            target_columns=target_columns,
            model_dir=model_dir,
            overrides=hparam_overrides,
            model_kwargs=model_kwargs,
            smiles_column=dataset_spec.smiles_column,
        )
        summary_rows.append(
            {
                "dataset": name,
                "num_tasks": metrics["num_targets"],
                "test_roc_auc": metrics["test_roc_auc"],
                "best_val_auc": metrics["best_val_auc"],
                "num_parameters": metrics["num_parameters"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )

    if not summary_rows:
        raise SystemExit(
            "No datasets were benchmarked. Download the data first (see commands.md)."
        )

    results_dir = ensure_dir(args.results_dir)
    summary_path = results_dir / "benchmark_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nBenchmark summary (test ROC-AUC)")
    for row in summary_rows:
        print(
            f"  {str(row['dataset']):<10} "
            f"tasks={row['num_tasks']:<3} "
            f"test_roc_auc={float(row['test_roc_auc']):.4f}"
        )
    if skipped:
        print(f"\nSkipped (missing data): {', '.join(skipped)}")
    print(f"\nSaved benchmark summary to: {summary_path}")
    print("Run `python results/main.py` to build per-dataset curves and the full results table.")


if __name__ == "__main__":
    main()
