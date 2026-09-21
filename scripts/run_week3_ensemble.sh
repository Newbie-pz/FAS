#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
ROOT="${ROOT:-outputs_week3_three_routes}"
OUT="${OUT:-outputs_week3_ensemble}"
GPU="${GPU:-0}"
TARGETS="${TARGETS:-CASIA MSU-MFSD Replay-Attack}"
BATCH_SIZE="${BATCH_SIZE:-32}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT"

echo "===================================================================="
echo "Week 3 DINO + SSDG ensemble"
echo "Targets      : $TARGETS"
echo "GPU          : $GPU"
echo "Batch size   : $BATCH_SIZE"
echo "Output       : $OUT"
echo "===================================================================="

for target in $TARGETS; do
  dino_ckpt="$ROOT/route2_dinov2_train/$target/best.pth"
  ssdg_ckpt="$ROOT/route3_ssdg_dino/$target/best.pth"

  if [[ ! -f "$dino_ckpt" ]]; then
    echo "[ERROR] Missing DINO checkpoint: $dino_ckpt" >&2
    exit 1
  fi
  if [[ ! -f "$ssdg_ckpt" ]]; then
    echo "[ERROR] Missing SSDG checkpoint: $ssdg_ckpt" >&2
    exit 1
  fi

  echo ""
  echo "===================================================================="
  echo "ENSEMBLE -> $target"
  echo "===================================================================="

  python evaluate_week3_ensemble.py     --dino_checkpoint "$dino_ckpt"     --ssdg_checkpoint "$ssdg_ckpt"     --dataset "$target"     --data_root "$DATA_ROOT"     --output_dir "$OUT"     --batch_size "$BATCH_SIZE"
done

python summarize_week3_ensemble.py --root "$OUT"

echo ""
echo "===================================================================="
echo "Ensemble evaluation completed."
echo "Summary: $OUT/ensemble_summary.md"
echo "===================================================================="
