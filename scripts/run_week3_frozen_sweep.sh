#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
GPU="${GPU:-0}"
BASE_ROOT="${BASE_ROOT:-outputs_week3_parallel_sweep}"
OUT_ROOT="${OUT_ROOT:-outputs_week3_frozen_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/week3_frozen_sweep}"
TARGETS="${TARGETS:-CASIA MSU-MFSD Replay-Attack}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
EPOCHS="${EPOCHS:-12}"
PATIENCE="${PATIENCE:-4}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

wait_for_slot() {
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
    wait -n || true
  done
}

run_baseline_tta() {
  local target="$1"
  local ckpt="$BASE_ROOT/dino_f12_lr5e6_train/$target/best.pth"
  local out="$OUT_ROOT/baseline_f12_tta_eval"
  local log="$LOG_ROOT/baseline_f12_tta_${target}.log"

  echo "[START] baseline_f12_tta -> $target"
  python evaluate_vfm.py     --checkpoint "$ckpt"     --dataset "$target"     --data_root "$DATA_ROOT"     --output_dir "$out"     --batch_size 64     --num_workers "$NUM_WORKERS"     --tta > "$log" 2>&1

  if [[ $? -eq 0 ]]; then
    echo "[DONE]  baseline_f12_tta -> $target"
  else
    echo "[FAIL]  baseline_f12_tta -> $target"
  fi
  return 0
}

run_variant() {
  local target="$1"
  local name="$2"
  local feature="$3"
  local loss="$4"
  local gamma="$5"
  local head_lr="$6"
  local dropout="$7"
  local smoothing="$8"

  local train_out="$OUT_ROOT/${name}_train"
  local eval_out="$OUT_ROOT/${name}_eval"
  local log="$LOG_ROOT/${name}_${target}.log"

  echo "[START] $name -> $target"
  {
    python train_vfm_multisource.py       --data_root "$DATA_ROOT"       --target_dataset "$target"       --output_dir "$train_out"       --epochs "$EPOCHS"       --batch_size "$BATCH_SIZE"       --grad_accum 1       --num_workers "$NUM_WORKERS"       --freeze_first_blocks 12       --feature_mode "$feature"       --dropout "$dropout"       --loss "$loss"       --focal_gamma "$gamma"       --label_smoothing "$smoothing"       --backbone_lr 1e-6       --head_lr "$head_lr"       --patience "$PATIENCE"       --max_train_frames_per_video 20       --max_val_frames_per_video 20

    python evaluate_vfm.py       --checkpoint "$train_out/$target/best.pth"       --dataset "$target"       --data_root "$DATA_ROOT"       --output_dir "$eval_out"       --batch_size 64       --num_workers "$NUM_WORKERS"       --tta
  } > "$log" 2>&1

  if [[ $? -eq 0 ]]; then
    echo "[DONE]  $name -> $target"
  else
    echo "[FAIL]  $name -> $target"
  fi
  return 0
}

echo "===================================================================="
echo "Week 3 frozen-DINO parallel sweep"
echo "Targets      : $TARGETS"
echo "Max parallel : $MAX_PARALLEL"
echo "Epochs       : $EPOCHS"
echo "Batch size   : $BATCH_SIZE"
echo "===================================================================="

for target in $TARGETS; do
  wait_for_slot
  run_baseline_tta "$target" &

  wait_for_slot
  run_variant "$target" "cls_focal_g1_lr1e4" cls focal 1.0 1e-4 0.2 0.0 &

  wait_for_slot
  run_variant "$target" "cls_focal_g2_lr1e4" cls focal 2.0 1e-4 0.2 0.0 &

  wait_for_slot
  run_variant "$target" "cls_focal_g2_lr2e4" cls focal 2.0 2e-4 0.2 0.0 &

  wait_for_slot
  run_variant "$target" "cls_ce_lr1e4" cls ce 2.0 1e-4 0.2 0.0 &

  wait_for_slot
  run_variant "$target" "clsmean_focal_g2_lr1e4" cls_mean focal 2.0 1e-4 0.2 0.0 &

  wait_for_slot
  run_variant "$target" "clsmeanstd_focal_g2_lr1e4" cls_mean_std focal 2.0 1e-4 0.2 0.0 &

  wait_for_slot
  run_variant "$target" "cls_focal_g2_lr1e4_ls005" cls focal 2.0 1e-4 0.2 0.05 &
done

wait

python summarize_week3_frozen_sweep.py --root "$OUT_ROOT" --targets CASIA MSU-MFSD Replay-Attack

echo "===================================================================="
echo "Frozen-DINO sweep completed."
echo "Summary: $OUT_ROOT/frozen_sweep_summary.md"
echo "===================================================================="
