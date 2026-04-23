from __future__ import annotations

import argparse
from pathlib import Path

from attentive_fp.main import MODEL_SPEC as ATTENTIVE_FP_SPEC
from common.config import get_project_root
from common.runner import add_shared_training_arguments, collect_override_hparams, run_experiment
from d_mpnn.main import MODEL_SPEC as D_MPNN_SPEC
from dual_kd_gnn.main import MODEL_SPEC as DUAL_KD_GNN_SPEC
from fp_gnn.main import MODEL_SPEC as FP_GNN_SPEC
from mlfgnn.main import MODEL_SPEC as MLFGNN_SPEC
from ml_mpnn.main import MODEL_SPEC as ML_MPNN_SPEC
from multichem.main import MODEL_SPEC as MULTICHEM_SPEC


ALL_SPECS = [
    ATTENTIVE_FP_SPEC,
    D_MPNN_SPEC,
    FP_GNN_SPEC,
    ML_MPNN_SPEC,
    MLFGNN_SPEC,
    MULTICHEM_SPEC,
    DUAL_KD_GNN_SPEC,
]
SPEC_BY_SLUG = {spec.slug: spec for spec in ALL_SPECS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one or more modularized molecular property prediction models.")
    add_shared_training_arguments(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model slugs to run. Use 'all' to train every model sequentially.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    requested_models = args.models
    if requested_models == ["all"] or "all" in requested_models:
        specs = ALL_SPECS
    else:
        unknown = [slug for slug in requested_models if slug not in SPEC_BY_SLUG]
        if unknown:
            parser.error(f"Unknown model slugs: {', '.join(unknown)}")
        specs = [SPEC_BY_SLUG[slug] for slug in requested_models]

    overrides = collect_override_hparams(args)
    project_root = get_project_root()
    summary = []
    for spec in specs:
        print(f"\n{'=' * 72}")
        print(f"Training model: {spec.name}")
        print(f"{'=' * 72}")
        metrics = run_experiment(
            spec=spec,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            seed=args.seed,
            device_name=args.device,
            target_columns=args.target_columns,
            model_dir=project_root / spec.slug,
            overrides=overrides,
        )
        summary.append((spec.name, metrics["test_roc_auc"]))

    print("\nFinal test ROC-AUC summary")
    for model_name, score in summary:
        print(f"  {model_name:<24} {score:.4f}")


if __name__ == "__main__":
    main()
