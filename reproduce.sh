#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="usl_project"
MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"

# ── 1. Ensure curl or wget is available ──────────────────────────────────────
if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
    echo "Neither curl nor wget found — attempting to install curl..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y curl
    elif command -v brew &>/dev/null; then
        brew install curl
    else
        echo "ERROR: cannot install curl automatically. Install curl or wget and retry."
        exit 1
    fi
fi

# ── 2. Find and initialise conda ─────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    for candidate in \
        "$HOME/anaconda3" \
        "$HOME/miniconda3" \
        "$HOME/miniforge3" \
        "$HOME/mambaforge" \
        "/opt/conda" \
        "/opt/anaconda3" \
        "/opt/miniconda3"
    do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            source "${candidate}/etc/profile.d/conda.sh"
            break
        fi
    done
fi

if ! command -v conda &>/dev/null; then
    echo "conda not found — installing Miniconda..."
    if command -v curl &>/dev/null; then
        curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "$MINICONDA_INSTALLER"
    else
        wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "$MINICONDA_INSTALLER"
    fi
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$MINICONDA_INSTALLER"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda init bash
    echo "Miniconda installed at $HOME/miniconda3"
fi

# ── 3. Create conda environment if needed ────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

# ── 4. Ensure Kaggle credentials are present ─────────────────────────────────
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ ! -f "$KAGGLE_JSON" ]; then
    mkdir -p "$KAGGLE_DIR"
    printf '{"username":"miguelteixeira7","key":"KGAT_433f59601314f5c99b8b52008db44af1"}' > "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
    echo "Kaggle credentials written to $KAGGLE_JSON"
fi

# ── 5. Run pipeline ───────────────────────────────────────────────────────────
echo "Running pipeline..."
conda run -n "${ENV_NAME}" --no-capture-output python run_all.py "$@"
