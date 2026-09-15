#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/Desktop/code/FAS/ProcessedData}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_data_audit}"

printf '%s\n' '=============================================='
printf '%s\n' 'ProcessedData 数据质量审计'
printf '%s\n' "数据目录: ${DATA_ROOT}"
printf '%s\n' "输出目录: ${OUTPUT_DIR}"
printf '%s\n' '检查内容: 可读性 / 标签 / 尺寸 / Group / frame 命名'
printf '%s\n' '=============================================='

python scripts/audit_processed_data.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}"

printf '\n数据审计完成。\n'
printf '报告: %s/processed_data_audit.md\n' "${OUTPUT_DIR}"
printf 'JSON: %s/processed_data_audit.json\n' "${OUTPUT_DIR}"
