from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [
    "attentive_fp",
    "d_mpnn",
    "fp_gnn",
    "ml_mpnn",
    "mlfgnn",
    "multichem",
    "dual_kd_gnn",
]


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for index, row in df.iterrows():
        rendered_row = [str(index)]
        for value in row.tolist():
            if pd.isna(value):
                rendered_row.append("")
            elif isinstance(value, float):
                rendered_row.append(format(value, floatfmt))
            else:
                rendered_row.append(str(value))
        rows.append(rendered_row)

    table_headers = [df.index.name or "index", *headers]
    separator = ["---"] * len(table_headers)
    lines = [
        "| " + " | ".join(table_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def collect_latest_runs():
    latest_runs: dict[tuple[str, str], dict[str, Path]] = {}
    for model_dir_name in MODEL_DIRS:
        model_dir = PROJECT_ROOT / model_dir_name
        for history_path in model_dir.glob("runs/*/training_log.csv"):
            dataset_name = history_path.parent.name
            metrics_path = history_path.with_name("metrics.json")
            if not metrics_path.exists():
                continue
            key = (model_dir_name, dataset_name)
            existing = latest_runs.get(key)
            if existing is None or history_path.stat().st_mtime > existing["history_path"].stat().st_mtime:
                latest_runs[key] = {
                    "model_dir": model_dir,
                    "history_path": history_path,
                    "metrics_path": metrics_path,
                }
    return latest_runs


def load_run_records():
    records = []
    for run_info in collect_latest_runs().values():
        metrics = json.loads(run_info["metrics_path"].read_text(encoding="utf-8"))
        history_df = pd.read_csv(run_info["history_path"])
        records.append(
            {
                "model_name": metrics["model_name"],
                "model_slug": metrics["model_slug"],
                "dataset_name": metrics["dataset_name"],
                "history": history_df,
                "metrics": metrics,
            }
        )
    return records


def plot_validation_auc_curves(records, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(record["dataset_name"], []).append(record)

    for dataset_name, dataset_records in grouped.items():
        plt.figure(figsize=(14, 8))
        for record in sorted(dataset_records, key=lambda item: item["model_name"]):
            history = record["history"]
            if "global_epoch" not in history.columns or "val_metric" not in history.columns:
                continue
            history = history.dropna(subset=["val_metric"])
            if history.empty:
                continue
            plt.plot(
                history["global_epoch"],
                history["val_metric"],
                linewidth=2,
                label=record["model_name"],
            )
        plt.xlabel("Epoch")
        plt.ylabel("Validation ROC-AUC")
        plt.title(f"Validation ROC-AUC Comparison ({dataset_name})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_val_auc_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_performance_tables(records, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = [record["metrics"] for record in records]
    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        return

    metrics_df = metrics_df.sort_values(["dataset_name", "model_name"]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)

    pivot = metrics_df.pivot(index="model_name", columns="dataset_name", values="test_roc_auc")
    pivot = pivot.sort_index()
    pivot.to_csv(output_dir / "performance_table.csv")
    (output_dir / "performance_table.md").write_text(
        dataframe_to_markdown(pivot, floatfmt=".4f"),
        encoding="utf-8",
    )


def main() -> None:
    records = load_run_records()
    if not records:
        raise SystemExit("No model run artifacts were found. Train at least one model first.")

    artifacts_dir = PROJECT_ROOT / "results" / "artifacts"
    plot_validation_auc_curves(records, artifacts_dir)
    save_performance_tables(records, artifacts_dir)
    print(f"Saved aggregated artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
