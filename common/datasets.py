from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.config import DEFAULT_TARGET_COLUMNS, get_project_root


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a MoleculeNet-style classification dataset.

    ``target_columns`` may be ``None`` to mean "every column except the SMILES
    column"; this is convenient for datasets such as SIDER whose only
    non-target column is the SMILES string.
    """

    name: str
    csv_filename: str
    smiles_column: str
    url: str
    target_columns: tuple[str, ...] | None = None
    description: str = ""

    def data_path(self, data_dir: Path | None = None) -> Path:
        data_dir = data_dir if data_dir is not None else (get_project_root() / "data")
        return data_dir / self.csv_filename


# DeepChem hosts the canonical MoleculeNet CSVs. ``.csv.gz`` files are gzipped;
# the downloader (scripts/download_data.py) decompresses them to plain CSV.
_S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

DATASETS: dict[str, DatasetSpec] = {
    "tox21": DatasetSpec(
        name="tox21",
        csv_filename="tox21.csv",
        smiles_column="smiles",
        url=f"{_S3}/tox21.csv.gz",
        target_columns=tuple(DEFAULT_TARGET_COLUMNS),
        description="Tox21 - 12 nuclear-receptor / stress-response toxicity assays (multitask binary).",
    ),
    "bbbp": DatasetSpec(
        name="bbbp",
        csv_filename="bbbp.csv",
        smiles_column="smiles",
        url=f"{_S3}/BBBP.csv",
        target_columns=("p_np",),
        description="BBBP - blood-brain barrier penetration (single binary task).",
    ),
    "bace": DatasetSpec(
        name="bace",
        csv_filename="bace.csv",
        smiles_column="mol",
        url=f"{_S3}/bace.csv",
        target_columns=("Class",),
        description="BACE - beta-secretase 1 (BACE-1) inhibition (single binary task).",
    ),
    "sider": DatasetSpec(
        name="sider",
        csv_filename="sider.csv",
        smiles_column="smiles",
        url=f"{_S3}/sider.csv.gz",
        target_columns=None,  # 27 side-effect system-organ-class tasks (all non-SMILES columns).
        description="SIDER - 27 marketed-drug adverse-reaction system-organ-class tasks (multitask binary).",
    ),
    "clintox": DatasetSpec(
        name="clintox",
        csv_filename="clintox.csv",
        smiles_column="smiles",
        url=f"{_S3}/clintox.csv.gz",
        target_columns=("FDA_APPROVED", "CT_TOX"),
        description="ClinTox - FDA approval status vs. clinical-trial toxicity (2 binary tasks).",
    ),
}

# Order used by the batch benchmark runner.
DEFAULT_BENCHMARK_DATASETS = ["bace", "bbbp", "sider", "tox21", "clintox"]
DEFAULT_DATASET = "tox21"


def available_datasets() -> list[str]:
    return list(DATASETS.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {', '.join(available_datasets())}."
        )
    return DATASETS[key]


def resolve_target_columns(spec: DatasetSpec, data_path: str | Path) -> list[str]:
    """Return the explicit target columns, or infer them from the CSV header."""
    if spec.target_columns is not None:
        return list(spec.target_columns)
    header = pd.read_csv(data_path, nrows=0)
    return [column for column in header.columns if column != spec.smiles_column]
