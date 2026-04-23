from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_roc_auc(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    num_classes: int,
) -> tuple[float, list[float]]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    aucs: list[float] = []
    for class_idx in range(num_classes):
        valid_mask = y_true[:, class_idx] != -1
        if int(valid_mask.sum()) == 0:
            continue
        true_values = y_true[valid_mask, class_idx]
        pred_values = y_pred[valid_mask, class_idx]
        if len(np.unique(true_values)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(true_values, pred_values)))
        except ValueError:
            continue

    return (float(np.mean(aucs)) if aucs else 0.0, aucs)


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float | list[float]]:
    probabilities = torch.sigmoid(outputs)
    mean_auc, per_task_aucs = compute_roc_auc(targets, probabilities, num_classes)
    return {
        "roc_auc": mean_auc,
        "roc_auc_per_task": per_task_aucs,
    }
