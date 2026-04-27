#!/usr/bin/env bash
set -e

PYTHON_BIN=${PYTHON_BIN:-python}
export MODEL_DIR=${MODEL_DIR:-models/story-gpt2}

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
	echo "Torch not available in this Python. Use your micromamba env or set PYTHON_BIN to it." >&2
	exit 1
fi

"$PYTHON_BIN" -m uvicorn api.app:app --host 0.0.0.0 --port 8000
