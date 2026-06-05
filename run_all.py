#!/usr/bin/env python
# run_all.py -- Regenerate all project results end-to-end.
#
# Usage:
#   python run_all.py              -- full run (all tasks + extensions)
#   python run_all.py --core-only  -- core tasks only
#
# Execution order:
#   1. src/project.ipynb                  -- Task 1: K-Means baseline
#   2. src/task2_gmm.ipynb                -- Task 2 + 3.2: GMM + AIC/BIC
#   3. src/task3_evaluation.ipynb         -- Task 3.1 + 3.3 + 3.4
#   4. src/extension_E2_spectral.ipynb    -- Extension E2: Spectral clustering
#   5. src/extension_E5_visualization.ipynb -- Extension E5: t-SNE

import argparse, shutil, subprocess, sys
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
from kagglehub.exceptions import DataCorruptionError

SRC_DIR = Path(__file__).parent / "src"

DATA_FILE = SRC_DIR / "hotel_bookings_course_release_v1.csv"


KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "jessemostipak" / "hotel-booking-demand"


def download_data():
    print("\n" + "="*60 + "\nDownloading dataset from Kaggle\n" + "="*60)
    try:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "jessemostipak/hotel-booking-demand",
            "hotel_bookings.csv",
        )
    except DataCorruptionError:
        print("Corrupted cache detected — clearing and retrying...")
        shutil.rmtree(KAGGLE_CACHE, ignore_errors=True)
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "jessemostipak/hotel-booking-demand",
            "hotel_bookings.csv",
        )
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved: {DATA_FILE}  ({len(df):,} rows)")
    print("First 5 records:", df.head())

CORE_NOTEBOOKS = [
    SRC_DIR / "project.ipynb",
    SRC_DIR / "task2_gmm.ipynb",
    SRC_DIR / "task3_evaluation.ipynb",
]

EXTENSION_NOTEBOOKS = [
    SRC_DIR / "extension_E2_spectral.ipynb",
    SRC_DIR / "extension_E5_visualization.ipynb",
]


def run_notebook(nb):
    print(f"\n" + "="*60 + f"\nRunning: {nb.name}\n" + "="*60)
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=7200", str(nb)],
        cwd=str(nb.parent),
    )
    if result.returncode != 0:
        print(f"ERROR: {nb.name} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"Done: {nb.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    if not DATA_FILE.exists():
        download_data()
    else:
        print(f"Dataset already present: {DATA_FILE}")

    notebooks = CORE_NOTEBOOKS if args.core_only else CORE_NOTEBOOKS + EXTENSION_NOTEBOOKS
    print(f"\nRunning {len(notebooks)} notebook(s)")
    for nb in notebooks:
        if nb.exists():
            run_notebook(nb)
        else:
            print(f"WARNING: {nb} not found")
    print("\n Done. All results in tables/ and figures/.")
