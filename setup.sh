#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${1:-}" ]]; then
    VENV_DIR="$1"
else
    read -rp "Enter path for the venv: " VENV_DIR
    if [[ -z "$VENV_DIR" ]]; then
        echo "Error: venv path cannot be empty." >&2
        exit 1
    fi
fi
PYTHON_VERSION="3.12"
TORCH_VERSION="2.10.0+cu128"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "==> Creating venv at $VENV_DIR"
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

echo "==> Activating venv"
source "$VENV_DIR/bin/activate"

echo "==> Installing packages from requirements.txt"
uv pip install -r requirements.txt --index-strategy unsafe-best-match

echo "==> Installing torch $TORCH_VERSION (cu128 wheel)"
uv pip install "torch==$TORCH_VERSION" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match

echo "==> Installing analysis packages (irrCAC, krippendorff) separately"
uv pip install irrCAC krippendorff tabulate --index-strategy unsafe-best-match

echo "==> Installing aiofiles separately"
uv pip install aiofiles --index-strategy unsafe-best-match

echo "==> Done. Activate with:"
echo "    source $VENV_DIR/bin/activate"
