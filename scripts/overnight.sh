#!/bin/bash
# Runs a full 50-epoch training pass, promotes best.pth to the production
# checkpoint key, and terminates the pod. Designed to be launched in tmux
# and left to run unattended:
#   tmux new -s overnight -d 'bash scripts/overnight.sh'
#
# Refuses to terminate the pod unless a real best.pth exists in S3, so a
# crash leaves the pod alive for debugging rather than wiping your work.

set -u
LOG=~/overnight.log
exec > >(tee -a "$LOG") 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Pulling latest code ==="
git pull --ff-only

echo "=== Preflight: chipped data ==="
if [ ! -d data/chipped/images ] || [ -z "$(ls -A data/chipped/images 2>/dev/null)" ]; then
  echo "FATAL: data/chipped/images empty — leaving pod alive"
  exit 1
fi

echo "=== Preflight: AWS creds ==="
aws s3 ls s3://golf-mapping-checkpoints/ >/dev/null \
  || { echo "FATAL: AWS creds broken — leaving pod alive"; exit 1; }

echo "=== Training start: $(date) ==="
timeout 6h python -m training.train \
  --data-dir data/chipped \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --output-dir s3://golf-mapping-checkpoints/models/
TRAIN_EXIT=$?
echo "=== Training ended with exit $TRAIN_EXIT at $(date) ==="

# Upload log regardless of outcome so we can always postmortem
aws s3 cp "$LOG" "s3://golf-mapping-checkpoints/models/overnight-$(date +%Y%m%d-%H%M).log" || true

if ! aws s3 ls s3://golf-mapping-checkpoints/models/best.pth >/dev/null 2>&1; then
  echo "FATAL: no best.pth in S3 — leaving pod alive for debugging"
  exit 1
fi

echo "=== Promoting best.pth to production key ==="
aws s3 cp s3://golf-mapping-checkpoints/models/best.pth \
          s3://golf-mapping-checkpoints/models/deeplabv3plus_v1.2.pth

echo "=== Terminating pod: $(date) ==="
sleep 5
runpodctl remove pod "$RUNPOD_POD_ID"
