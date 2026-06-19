"""Download the benchmark datasets (BACE, BBBP, SIDER, Tox21, ClinTox).

Examples
--------
    python scripts/download_data.py            # download every dataset
    python scripts/download_data.py all        # same as above
    python scripts/download_data.py bbbp bace  # download a subset
    python scripts/download_data.py --force tox21

Gzipped sources (``.csv.gz``) are decompressed to plain ``.csv`` under data/.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import DATASETS, available_datasets, get_dataset_spec  # noqa: E402


def download_one(name: str, data_dir: Path, force: bool) -> None:
    spec = get_dataset_spec(name)
    destination = data_dir / spec.csv_filename
    if destination.exists() and not force:
        print(f"[skip] {name}: already present at {destination} (use --force to re-download).")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {name}: {spec.url}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(spec.url, tmp_path)
        if spec.url.endswith(".gz"):
            with gzip.open(tmp_path, "rb") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"[done] {name}: saved to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets into data/.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        help=f"Datasets to download (default: all). Choices: {', '.join(available_datasets())}, all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Destination directory (default: ./data).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()

    requested = args.datasets or ["all"]
    names = list(DATASETS.keys()) if "all" in requested else requested
    unknown = [name for name in names if name.lower() not in DATASETS]
    if unknown:
        parser.error(f"Unknown dataset(s): {', '.join(unknown)}. Choices: {', '.join(available_datasets())}, all.")

    for name in names:
        download_one(name, args.data_dir, args.force)


if __name__ == "__main__":
    main()
