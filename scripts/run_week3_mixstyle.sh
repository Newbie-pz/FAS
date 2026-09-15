#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}
SOURCE_DATASET=${SOURCE_DATASET:-OULU-NPU}
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-outputs_week3/mixstyle}
CROSS_OUTPUT_ROOT=${CROSS_OUTPUT_ROOT:-outputs_week3_cross/mixstyle}
TARGET_DATASETS=(CASIA MSU-MFSD Replay-Attack)

printf '%s\n' '=============================================='
printf '%s\n' '第三周 MixStyle 域泛化实验'
printf '源数据集      : %s\n' "$SOURCE_DATASET"
printf '方法          : mixstyle\n'
printf '训练输出目录  : %s\n' "$TRAIN_OUTPUT_ROOT"
printf '跨域输出目录  : %s\n' "$CROSS_OUTPUT_ROOT"
printf '目标数据集    : %s\n' "${TARGET_DATASETS[*]}"
printf '%s\n\n' '=============================================='

python train.py \
  --dataset "$SOURCE_DATASET" \
  --data_root "$DATA_ROOT" \
  --output_dir "$TRAIN_OUTPUT_ROOT" \
  --augmentation baseline \
  --method mixstyle

CHECKPOINT="$TRAIN_OUTPUT_ROOT/$SOURCE_DATASET/best.pth"

for target in "${TARGET_DATASETS[@]}"; do
  printf '\n%s\n' '----------------------------------------------'
  printf '开始测试: %s -> %s\n' "$SOURCE_DATASET" "$target"
  printf '%s\n\n' '----------------------------------------------'

  python evaluate_cross_dataset.py \
    --checkpoint "$CHECKPOINT" \
    --source_dataset "$SOURCE_DATASET" \
    --dataset "$target" \
    --data_root "$DATA_ROOT" \
    --output_dir "$CROSS_OUTPUT_ROOT"
done

python summarize_cross_results.py \
  --root "$CROSS_OUTPUT_ROOT" \
  --csv_name mixstyle_cross_dataset_summary.csv \
  --md_name mixstyle_cross_dataset_summary.md

python compare_week3_results.py \
  --candidate "$CROSS_OUTPUT_ROOT/mixstyle_cross_dataset_summary.csv" \
  --method_name MixStyle \
  --output "$CROSS_OUTPUT_ROOT/week3_comparison.md"

printf '\nMixStyle 域泛化实验完成。\n'
