#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=============================================='
printf '%s\n' '第四周实验对比与可视化分析'
printf '%s\n' '对比方法: Baseline / Strong / Appearance'
printf '%s\n\n' '=============================================='

python analyze_week4.py

printf '\n第四周分析完成。结果目录: outputs_week4/\n'
