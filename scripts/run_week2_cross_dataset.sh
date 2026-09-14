#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
SOURCE="${SOURCE:-OULU-NPU}"
CHECKPOINT="${CHECKPOINT:-outputs/${SOURCE}/best.pth}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TARGETS=("CASIA" "MSU-MFSD" "Replay-Attack")

echo "=============================================="
echo "第二周跨数据集实验"
echo "源数据集      : ${SOURCE}"
echo "Checkpoint    : ${CHECKPOINT}"
echo "数据根目录    : ${DATA_ROOT}"
echo "目标数据集    : ${TARGETS[*]}"
echo "=============================================="

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[错误] 未找到 checkpoint: ${CHECKPOINT}" >&2
  echo "请先完成第一周源数据集训练。" >&2
  exit 1
fi

for TARGET in "${TARGETS[@]}"; do
  echo
  echo "----------------------------------------------"
  echo "开始测试: ${SOURCE} -> ${TARGET}"
  echo "----------------------------------------------"
  python evaluate_cross_dataset.py \
    --checkpoint "${CHECKPOINT}" \
    --source_dataset "${SOURCE}" \
    --dataset "${TARGET}" \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
done

python summarize_cross_results.py --root outputs_cross

echo
echo "第二周跨数据集实验完成。"
