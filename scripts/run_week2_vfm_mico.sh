#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
TRAIN_OUT="${TRAIN_OUT:-outputs_week2_vfm}"
EVAL_OUT="${EVAL_OUT:-outputs_week2_vfm_eval}"
GPU="${GPU:-0}"
TARGETS="${TARGETS:-CASIA MSU-MFSD Replay-Attack}"
MODEL_NAME="${MODEL_NAME:-dinov2_vitb14_reg}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
BACKBONE_LR="${BACKBONE_LR:-5e-6}"
HEAD_LR="${HEAD_LR:-5e-5}"
MAX_FRAMES="${MAX_FRAMES:-20}"
FREEZE_FIRST_BLOCKS="${FREEZE_FIRST_BLOCKS:-11}"
FEATURE_MODE="${FEATURE_MODE:-cls}"
LOSS="${LOSS:-focal}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
PATIENCE="${PATIENCE:-15}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "Week2 high-performance VFM / MICO"
echo "GPU              : $GPU"
echo "Targets          : $TARGETS"
echo "Model            : $MODEL_NAME"
echo "Epochs           : $EPOCHS"
echo "Batch            : $BATCH_SIZE"
echo "Gradient accum.  : $GRAD_ACCUM"
echo "Backbone LR      : $BACKBONE_LR"
echo "Head LR          : $HEAD_LR"
echo "Frames/video     : $MAX_FRAMES"
echo "Freeze blocks    : $FREEZE_FIRST_BLOCKS"
echo "Feature mode     : $FEATURE_MODE"
echo "Loss             : $LOSS"
echo "Patience         : $PATIENCE"
echo "============================================================"

for target in $TARGETS; do
  echo ""
  echo "============================================================"
  echo "[TRAIN] unseen target: $target"
  echo "============================================================"

  python train_vfm_multisource.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$TRAIN_OUT"     --model_name "$MODEL_NAME"     --epochs "$EPOCHS"     --batch_size "$BATCH_SIZE"     --grad_accum "$GRAD_ACCUM"     --backbone_lr "$BACKBONE_LR"     --head_lr "$HEAD_LR"     --max_train_frames_per_video "$MAX_FRAMES"     --max_val_frames_per_video "$MAX_FRAMES"     --feature_mode "$FEATURE_MODE"     --freeze_first_blocks "$FREEZE_FIRST_BLOCKS"     --loss "$LOSS"     --focal_gamma "$FOCAL_GAMMA"     --patience "$PATIENCE"

  echo ""
  echo "============================================================"
  echo "[EVAL] unseen target: $target"
  echo "============================================================"

  python evaluate_vfm.py     --checkpoint "$TRAIN_OUT/$target/best.pth"     --dataset "$target"     --data_root "$DATA_ROOT"     --output_dir "$EVAL_OUT"     --batch_size 64
done

python summarize_vfm_results.py --root "$EVAL_OUT"

echo ""
echo "============================================================"
echo "Week2 VFM / MICO completed."
echo "Summary: $EVAL_OUT/vfm_cross_domain_summary.md"
echo "============================================================"
