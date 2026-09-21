#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-outputs_week3_parallel_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/week3_parallel_sweep}"
TARGETS="${TARGETS:-CASIA MSU-MFSD Replay-Attack}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_ssdg() {
  local target="$1"
  local name="$2"
  local freeze="$3"
  local ad="$4"
  local tri="$5"

  local out="$OUT_ROOT/$name"
  local log="$LOG_ROOT/${name}_${target}.log"

  echo "[START] $name -> $target"
  python train_week3_ssdg_dino.py     --data_root "$DATA_ROOT"     --target_dataset "$target"     --output_dir "$out"     --epochs "$EPOCHS"     --batch_size "$BATCH_SIZE"     --num_workers "$NUM_WORKERS"     --freeze_first_blocks "$freeze"     --backbone_lr 2e-6     --head_lr 5e-5     --disc_lr 5e-5     --lambda_triplet "$tri"     --lambda_adreal "$ad"     --triplet_margin 0.1     --max_frames_per_video 20     --patience "$PATIENCE"     > "$log" 2>&1

  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "[DONE]  $name -> $target"
  else
    echo "[FAIL]  $name -> $target (exit=$code)"
  fi
  return 0
}

run_dino() {
  local target="$1"
  local name="$2"
  local freeze="$3"
  local blr="$4"

  local train_out="$OUT_ROOT/${name}_train"
  local eval_out="$OUT_ROOT/${name}_eval"
  local log="$LOG_ROOT/${name}_${target}.log"

  echo "[START] $name -> $target"
  {
    python train_vfm_multisource.py       --data_root "$DATA_ROOT"       --target_dataset "$target"       --output_dir "$train_out"       --epochs "$EPOCHS"       --batch_size "$BATCH_SIZE"       --grad_accum 2       --num_workers "$NUM_WORKERS"       --freeze_first_blocks "$freeze"       --feature_mode cls       --loss focal       --focal_gamma 2.0       --backbone_lr "$blr"       --head_lr 5e-5       --patience "$PATIENCE"       --max_train_frames_per_video 20       --max_val_frames_per_video 20

    python evaluate_vfm.py       --checkpoint "$train_out/$target/best.pth"       --dataset "$target"       --data_root "$DATA_ROOT"       --output_dir "$eval_out"       --batch_size 64       --num_workers "$NUM_WORKERS"
  } > "$log" 2>&1

  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "[DONE]  $name -> $target"
  else
    echo "[FAIL]  $name -> $target (exit=$code)"
  fi
  return 0
}

wait_for_slot() {
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
    wait -n || true
  done
}

echo "===================================================================="
echo "Week 3 parallel sweep"
echo "GPU          : $GPU"
echo "Targets      : $TARGETS"
echo "Max parallel : $MAX_PARALLEL"
echo "Epochs       : $EPOCHS"
echo "Patience     : $PATIENCE"
echo "Batch size   : $BATCH_SIZE"
echo "===================================================================="
echo "Variants:"
echo "  dino_f12_lr5e6"
echo "  dino_f10_lr2e6"
echo "  ssdg_f11_ad02_tri10"
echo "  ssdg_f11_ad01_tri05"
echo "  ssdg_f10_ad02_tri05"
echo "  ssdg_f12_ad02_tri05"
echo "===================================================================="

for target in $TARGETS; do
  wait_for_slot
  run_dino "$target" "dino_f12_lr5e6" 12 5e-6 &

  wait_for_slot
  run_dino "$target" "dino_f10_lr2e6" 10 2e-6 &

  wait_for_slot
  run_ssdg "$target" "ssdg_f11_ad02_tri10" 11 0.2 1.0 &

  wait_for_slot
  run_ssdg "$target" "ssdg_f11_ad01_tri05" 11 0.1 0.5 &

  wait_for_slot
  run_ssdg "$target" "ssdg_f10_ad02_tri05" 10 0.2 0.5 &

  wait_for_slot
  run_ssdg "$target" "ssdg_f12_ad02_tri05" 12 0.2 0.5 &
done

wait

python summarize_week3_parallel_sweep.py   --root "$OUT_ROOT"   --targets CASIA MSU-MFSD Replay-Attack

echo "===================================================================="
echo "Parallel sweep completed."
echo "Summary: $OUT_ROOT/parallel_sweep_summary.md"
echo "===================================================================="
