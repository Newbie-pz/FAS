#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
OUT_ROOT="${OUT_ROOT:-outputs_week3_three_routes}"
GPU="${GPU:-0}"
TARGETS="${TARGETS:-CASIA MSU-MFSD Replay-Attack}"

RESNET_EPOCHS="${RESNET_EPOCHS:-20}"
DINO_EPOCHS="${DINO_EPOCHS:-25}"
SSDG_EPOCHS="${SSDG_EPOCHS:-20}"
TDSF_EPOCHS="${TDSF_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_FRAMES="${MAX_FRAMES:-20}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_ROOT"

echo "===================================================================="
echo "第三周统一实验套件"
echo "方向 1: Multi-source DG + ResNet18"
echo "方向 2: DINOv2-Reg + Multi-source DG"
echo "方向 3A: DINOv2-Reg + SSDG-style"
echo "方向 3B: FAS-TD-SF-inspired temporal/spatial-gradient"
echo "Targets          : $TARGETS"
echo "GPU              : $GPU"
echo "Batch size       : $BATCH_SIZE"
echo "Frames/video     : $MAX_FRAMES"
echo "Output root      : $OUT_ROOT"
echo "===================================================================="

for target in $TARGETS; do
  echo ""
  echo "####################################################################"
  echo "TARGET = $target"
  echo "####################################################################"

  echo ""
  echo "===================================================================="
  echo "[1/4] Multi-source DG + ResNet18 -> $target"
  echo "===================================================================="
  python train_week3_multisource_resnet.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$OUT_ROOT/route1_resnet"     --epochs "$RESNET_EPOCHS"     --batch_size "$BATCH_SIZE"     --max_frames_per_video "$MAX_FRAMES"     --lr 1e-4     --weight_decay 1e-4     --patience 8

  echo ""
  echo "===================================================================="
  echo "[2/4] DINOv2-Reg + Multi-source DG -> $target"
  echo "===================================================================="
  python train_vfm_multisource.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$OUT_ROOT/route2_dinov2_train"     --epochs "$DINO_EPOCHS"     --batch_size "$BATCH_SIZE"     --grad_accum 2     --backbone_lr 5e-6     --head_lr 5e-5     --max_train_frames_per_video "$MAX_FRAMES"     --max_val_frames_per_video "$MAX_FRAMES"     --feature_mode cls     --freeze_first_blocks 11     --loss focal     --focal_gamma 2.0     --patience 10

  python evaluate_vfm.py     --checkpoint "$OUT_ROOT/route2_dinov2_train/$target/best.pth"     --dataset "$target"     --data_root "$DATA_ROOT"     --output_dir "$OUT_ROOT/route2_dinov2_eval"     --batch_size 64

  echo ""
  echo "===================================================================="
  echo "[3/4] DINOv2-Reg + SSDG-style -> $target"
  echo "===================================================================="
  python train_week3_ssdg_dino.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$OUT_ROOT/route3_ssdg_dino"     --epochs "$SSDG_EPOCHS"     --batch_size "$BATCH_SIZE"     --freeze_first_blocks 10     --backbone_lr 2e-6     --head_lr 5e-5     --disc_lr 5e-5     --lambda_triplet 1.0     --lambda_adreal 0.5     --triplet_margin 0.1     --max_frames_per_video "$MAX_FRAMES"     --patience 8

  echo ""
  echo "===================================================================="
  echo "[4/4] FAS-TD-SF-inspired -> $target"
  echo "===================================================================="
  python week3_td_sf_inspired.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$OUT_ROOT/route3_td_sf"     --eval_dir "$OUT_ROOT/route3_td_sf_eval"     --epochs "$TDSF_EPOCHS"     --batch_size 12     --freeze_first_blocks 11     --backbone_lr 3e-6     --head_lr 5e-5     --max_sequences_per_video 10     --patience 8

done

echo ""
echo "===================================================================="
echo "Summarizing Week 3 results"
echo "===================================================================="
python summarize_week3_three_routes.py --root "$OUT_ROOT"

echo ""
echo "===================================================================="
echo "Week 3 unified suite completed."
echo "Summary: $OUT_ROOT/week3_three_routes_summary.md"
echo "===================================================================="
