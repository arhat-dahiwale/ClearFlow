#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON=".venv/Scripts/python.exe"

# Verify Python 3.12 is available.
if ! command -v py >/dev/null 2>&1 ||
   ! py -3.12 -c \
       'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' \
       >/dev/null 2>&1; then
    echo "Error: Python 3.12 is required."
    echo "Install Python 3.12 and verify with: py -3.12 --version"
    exit 1
fi

# Refuse to reuse a broken environment.
if [[ -e ".venv" && ! -f "$VENV_PYTHON" ]]; then
    echo "Error: .venv exists but is not a valid Windows Python environment."
    echo "Inspect or remove .venv before rerunning setup."
    exit 1
fi

# Refuse to use an environment with the wrong Python version.
if [[ -f "$VENV_PYTHON" ]] &&
   ! "$VENV_PYTHON" -c \
       'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    echo "Error: .venv does not use Python 3.12."
    exit 1
fi

# Create the virtual environment if necessary.
if [[ ! -f "$VENV_PYTHON" ]]; then
    py -3.12 -m venv .venv
fi

# Create the expected directories on every teammate's machine.
mkdir -p \
    Datasets/CICIDS2017 \
    Datasets/SDN_TCP_SYN \
    data/interim \
    data/processed \
    notebooks \
    src \
    tests \
    results/figures \
    results/metrics \
    results/shap \
    docs/figures

# Prefer the complete lock file when the repository has one.
DEPENDENCIES="requirements.txt"

if [[ -f "requirements-lock.txt" ]]; then
    DEPENDENCIES="requirements-lock.txt"
fi

echo "Installing dependencies from $DEPENDENCIES..."

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$DEPENDENCIES"
"$VENV_PYTHON" -m pip check

# Verify essential imports.
"$VENV_PYTHON" -c \
    "import numpy, pandas, matplotlib, seaborn, sklearn, imblearn, joblib, shap, psutil, ipykernel; print('All required imports succeeded.')"

echo
echo "ClearFlow setup completed successfully."
echo "Activate with: source .venv/Scripts/activate"