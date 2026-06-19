from __future__ import annotations

from pathlib import Path
from typing import Iterable

from common.io_utils import ensure_dir


def _has_values(series) -> bool:
    return any(value is not None and value == value for value in series)  # value == value drops NaN


def plot_training_curves(
    history_rows: Iterable[dict[str, object]],
    output_path: Path,
    title: str,
) -> Path | None:
    """Save loss and ROC-AUC training curves for a single run.

    Returns the written PNG path, or ``None`` when matplotlib is unavailable
    or there is nothing to plot.
    """
    rows = list(history_rows)
    if not rows:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is best-effort.
        return None

    epochs = [row.get("global_epoch") for row in rows]
    train_loss = [row.get("train_loss") for row in rows]
    val_loss = [row.get("val_loss") for row in rows]
    train_metric = [row.get("train_metric") for row in rows]
    val_metric = [row.get("val_metric") for row in rows]

    ensure_dir(output_path.parent)
    fig, (loss_ax, auc_ax) = plt.subplots(1, 2, figsize=(14, 5))

    if _has_values(train_loss):
        loss_ax.plot(epochs, train_loss, label="Train loss", linewidth=2)
    if _has_values(val_loss):
        loss_ax.plot(epochs, val_loss, label="Val loss", linewidth=2)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title(f"{title} - loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    if _has_values(train_metric):
        auc_ax.plot(epochs, train_metric, label="Train ROC-AUC", linewidth=2)
    if _has_values(val_metric):
        auc_ax.plot(epochs, val_metric, label="Val ROC-AUC", linewidth=2)
    auc_ax.set_xlabel("Epoch")
    auc_ax.set_ylabel("ROC-AUC")
    auc_ax.set_title(f"{title} - ROC-AUC")
    auc_ax.grid(alpha=0.3)
    auc_ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
