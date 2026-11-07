
#!/usr/bin/env bash
set -euo pipefail

export QDRANT__STORAGE__STORAGE_PATH=/root/finrag-storage/qdrant-server
export QDRANT__SERVICE__HOST=127.0.0.1

cd /root/qdrant-server

exec ./qdrant
