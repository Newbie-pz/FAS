#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-outputs_week3_ensemble}"
OUT="${OUT:-outputs_week3_dynamic_fusion}"

python evaluate_week3_dynamic_fusion.py   --root "$ROOT"   --output "$OUT"   --targets CASIA MSU-MFSD Replay-Attack
