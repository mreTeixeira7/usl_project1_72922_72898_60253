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
#   4. src/task4_repository.ipynb         -- Task 4.2: repo deliverables
#   5. src/extension_E2_spectral.ipynb    -- Extension E2: Spectral clustering
#   6. src/extension_E5_visualization.ipynb -- Extension E5: t-SNE

import argparse, subprocess, sys
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"

CORE_NOTEBOOKS = [
    SRC_DIR / "project.ipynb",
    SRC_DIR / "task2_gmm.ipynb",
    SRC_DIR / "task3_evaluation.ipynb",
    SRC_DIR / "task4_repository.ipynb",
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
    notebooks = CORE_NOTEBOOKS if args.core_only else CORE_NOTEBOOKS + EXTENSION_NOTEBOOKS
    print(f"Running {len(notebooks)} notebook(s)")
    for nb in notebooks:
        if nb.exists():
            run_notebook(nb)
        else:
            print(f"WARNING: {nb} not found")
    print("\n Done. All results in tables/ and figures/.")
