#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
TRAIN_OUT="${TRAIN_OUT:-outputs_week2_vfm_strong}"
EVAL_OUT="${EVAL_OUT:-outputs_week2_vfm_strong_eval}"
GPU="${GPU:-0}"
TARGETS="${TARGETS:-CASIA}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
BACKBONE_LR="${BACKBONE_LR:-2e-6}"
HEAD_LR="${HEAD_LR:-5e-5}"
FREEZE_FIRST_BLOCKS="${FREEZE_FIRST_BLOCKS:-10}"
FAS_AUG_P="${FAS_AUG_P:-0.75}"
PDA_P="${PDA_P:-0.5}"
PDA_RATIO="${PDA_RATIO:-0.25}"
APL_WEIGHT="${APL_WEIGHT:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-20}"
PATIENCE="${PATIENCE:-8}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "Strong VFM: DINOv2-Reg + FAS-Aug + PDA + APL"
echo "GPU              : $GPU"
echo "Targets          : $TARGETS"
echo "Epochs           : $EPOCHS"
echo "Batch / accum    : $BATCH_SIZE / $GRAD_ACCUM"
echo "Backbone LR      : $BACKBONE_LR"
echo "Head LR          : $HEAD_LR"
echo "Freeze blocks    : $FREEZE_FIRST_BLOCKS"
echo "FAS-Aug p        : $FAS_AUG_P"
echo "PDA p / ratio    : $PDA_P / $PDA_RATIO"
echo "APL weight       : $APL_WEIGHT"
echo "Frames/video     : $MAX_FRAMES"
echo "============================================================"

for target in $TARGETS; do
  echo ""
  echo "============================================================"
  echo "[TRAIN] unseen target: $target"
  echo "============================================================"

  python train_vfm_patch.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$TRAIN_OUT"     --epochs "$EPOCHS"     --batch_size "$BATCH_SIZE"     --grad_accum "$GRAD_ACCUM"     --backbone_lr "$BACKBONE_LR"     --head_lr "$HEAD_LR"     --freeze_first_blocks "$FREEZE_FIRST_BLOCKS"     --fas_aug_p "$FAS_AUG_P"     --pda_p "$PDA_P"     --pda_replace_ratio "$PDA_RATIO"     --apl_weight "$APL_WEIGHT"     --max_train_frames_per_video "$MAX_FRAMES"     --max_val_frames_per_video "$MAX_FRAMES"     --patience "$PATIENCE"

  echo ""
  echo "============================================================"
  echo "[EVAL] unseen target: $target"
  echo "============================================================"

  python evaluate_vfm_patch.py     --checkpoint "$TRAIN_OUT/$target/best.pth"     --dataset "$target"     --data_root "$DATA_ROOT"     --output_dir "$EVAL_OUT"     --batch_size 64
done

echo ""
echo "============================================================"
echo "Strong VFM completed."
echo "Results under: $EVAL_OUT"
echo "============================================================"
