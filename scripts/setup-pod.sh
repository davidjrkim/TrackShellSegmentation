#!/bin/bash
# Bootstraps a fresh RunPod training instance.
# Idempotent — safe to re-run if something fails mid-way.
#
# Prerequisites (set these in your shell BEFORE running this script):
#   export AWS_REGION=ap-northeast-2
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export RUNPOD_API_KEY=...           # only needed if you'll run overnight.sh
#
# Usage:
#   cd /workspace && git clone https://github.com/davidjrkim/TrackShellSegmentation.git
#   cd TrackShellSegmentation
#   bash scripts/setup-pod.sh

set -euo pipefail

DATASET_S3="s3://golf-mapping-checkpoints/datasets/danish-golf-courses-orthophotos.tar"
DATASET_DIR_NAME="danish-golf-courses-orthophotos-DatasetNinja"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "==> Verifying AWS credentials"
: "${AWS_REGION:?AWS_REGION is not set}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is not set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is not set}"
aws s3 ls s3://golf-mapping-checkpoints/ >/dev/null

echo "==> Pulling latest code"
git pull --ff-only

echo "==> Installing training-only deps"
pip install --quiet \
  boto3==1.34.69 \
  tensorboard==2.16.2 \
  albumentations==1.4.3

if [ ! -d "$DATASET_DIR_NAME" ]; then
  echo "==> Downloading dataset from $DATASET_S3"
  aws s3 cp "$DATASET_S3" /tmp/dataset.tar
  tar -xf /tmp/dataset.tar
  rm /tmp/dataset.tar
fi

if [ ! -d data/chipped/images ] || [ -z "$(ls -A data/chipped/images 2>/dev/null)" ]; then
  echo "==> Chipping dataset (takes 5-15 min)"
  python -m training.prepare_data \
    --dataset-dir "$DATASET_DIR_NAME" \
    --out-dir data/chipped
fi

CHIP_COUNT=$(ls data/chipped/images | wc -l)
echo ""
echo "==> Ready. $CHIP_COUNT chips in data/chipped/images"
echo ""
echo "Next step — launch overnight training:"
echo "  [ ] export RUNPOD_API_KEY=rpa_...  (must be set)"
echo "  [ ] echo \$RUNPOD_POD_ID           (must be non-empty)"
echo "  [ ] tmux new -s overnight -d 'bash scripts/overnight.sh'"
