#!/bin/bash
set -e

# Check UV
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Build UV
uv venv
source .venv/bin/activate
export PYTHONPATH=.

# Install torch first, then project
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .