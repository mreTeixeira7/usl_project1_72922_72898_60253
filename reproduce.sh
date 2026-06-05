#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="usl_project"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

echo "Running pipeline..."
conda run -n "${ENV_NAME}" --no-capture-output python run_all.py "$@"
