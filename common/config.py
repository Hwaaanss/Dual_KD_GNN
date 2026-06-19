from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


DEFAULT_SEED = 42
DEFAULT_DATASET_NAME = "tox21"
DEFAULT_TARGET_COLUMNS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
NUM_CLASSES = len(DEFAULT_TARGET_COLUMNS)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_mps_available() -> bool:
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def get_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested_device}' but CUDA is not available. "
                "Refusing to silently fall back to CPU. "
                "Pass --device cpu explicitly to run on CPU."
            )
        if requested_device == "mps" and not is_mps_available():
            raise RuntimeError(
                "Requested device 'mps' but MPS is not available. "
                "Pass --device cpu explicitly to run on CPU."
            )
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    uses_dual_features: bool
    builder: Callable[..., torch.nn.Module]
    default_hparams: dict[str, Any] = field(default_factory=dict)
    add_model_arguments: Callable[[Any], None] | None = None
    collect_model_kwargs: Callable[[Any], dict[str, Any]] | None = None
    collect_hparam_overrides: Callable[[Any], dict[str, Any]] | None = None
    notes: str = ""
