
#!/usr/bin/env bash
set -euo pipefail

# FinRAG RunPod environment setup

cd /root/finrag

# 1. Local caches
mkdir -p /root/.cache/uv
mkdir -p /root/.cache/huggingface

export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# 2. Install Python dependencies
uv sync --locked --all-groups

# 3. Verify GPU
uv run --all-groups python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"

# 4. Download Qdrant if missing
QDRANT_DIR=/root/qdrant-server
QDRANT_VERSION=v1.20.0

mkdir -p "$QDRANT_DIR"
mkdir -p /root/finrag-storage/qdrant-server

if [ ! -x "$QDRANT_DIR/qdrant" ]; then
    curl -fL \
      "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" \
      -o /root/qdrant.tar.gz

    tar -xzf /root/qdrant.tar.gz -C "$QDRANT_DIR"
    chmod +x "$QDRANT_DIR/qdrant"
fi

echo "FinRAG environment ready!"
