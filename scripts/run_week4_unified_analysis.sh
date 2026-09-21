#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

THREE_ROUTES="${THREE_ROUTES:-outputs_week3_three_routes}"
ENSEMBLE_ROOT="${ENSEMBLE_ROOT:-outputs_week3_ensemble}"
DYNAMIC_ROOT="${DYNAMIC_ROOT:-outputs_week3_dynamic_fusion}"
PARALLEL_ROOT="${PARALLEL_ROOT:-outputs_week3_parallel_sweep}"
OUT="${OUT:-outputs_week4_analysis}"

echo "===================================================================="
echo "Week 4 unified analysis"
echo "No training will be performed."
echo "Final Week-3 model: Frozen DINOv2-Reg, mean Video AUC = 84.37%"
echo "===================================================================="

required=(
  "$PARALLEL_ROOT/dino_f12_lr5e6_eval/CASIA/vfm_summary.json"
  "$PARALLEL_ROOT/dino_f12_lr5e6_eval/MSU-MFSD/vfm_summary.json"
  "$PARALLEL_ROOT/dino_f12_lr5e6_eval/Replay-Attack/vfm_summary.json"
  "$PARALLEL_ROOT/parallel_sweep_summary.md"
  "$THREE_ROUTES/week3_three_routes_summary.md"
  "$DYNAMIC_ROOT/dynamic_fusion_summary.md"
)

for p in "${required[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing required result: $p" >&2
    exit 1
  fi
done

python week4_analysis.py   --three_routes "$THREE_ROUTES"   --ensemble_root "$ENSEMBLE_ROOT"   --dynamic_root "$DYNAMIC_ROOT"   --parallel_root "$PARALLEL_ROOT"   --output "$OUT"

echo ""
echo "===================================================================="
echo "Week 4 analysis completed."
echo "Summary : $OUT/week4_summary.md"
echo "Table   : $OUT/week4_method_comparison.csv"
echo "Figures : $OUT/*.png"
echo "===================================================================="
