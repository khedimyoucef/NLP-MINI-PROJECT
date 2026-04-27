#!/usr/bin/env bash
set -e

REPO_ID=${REPO_ID:-khedim/NLP-MINI-PROJECT}
MODEL_DIR=${MODEL_DIR:-models/story-gpt2}
PYTHON_BIN=${PYTHON_BIN:-python}
HF_TOKEN=${HF_TOKEN:-}

"$PYTHON_BIN" -m pip install -q -U huggingface_hub

HF_BIN=${HF_BIN:-"$(dirname "$PYTHON_BIN")/hf"}

TOKEN_ARGS=()
if [ -n "$HF_TOKEN" ]; then
	TOKEN_ARGS=(--token "$HF_TOKEN")
fi

if [ -x "$HF_BIN" ]; then
	"$HF_BIN" download "$REPO_ID" --repo-type model --local-dir "$MODEL_DIR" "${TOKEN_ARGS[@]}"
elif command -v hf >/dev/null 2>&1; then
	hf download "$REPO_ID" --repo-type model --local-dir "$MODEL_DIR" "${TOKEN_ARGS[@]}"
else
	"$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO_ID', repo_type='model', local_dir='$MODEL_DIR', local_dir_use_symlinks=False, token='$HF_TOKEN' if '$HF_TOKEN' else None)"
fi

echo "Downloaded model to $MODEL_DIR"