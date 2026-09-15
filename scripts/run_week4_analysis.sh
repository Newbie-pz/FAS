#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=============================================='
printf '%s\n' '第四周实验对比、消融与可视化分析'
printf '%s\n' '对比方法:'
printf '%s\n' '  1) Baseline'
printf '%s\n' '  2) Strong Augmentation'
printf '%s\n' '  3) Appearance Augmentation'
printf '%s\n' '  4) MixStyle'
printf '%s\n' '  5) Fourier Amplitude Augmentation'
printf '%s\n\n' '=============================================='

python analyze_week4.py

printf '\n第四周分析完成。\n'
printf '结果目录: outputs_week4/\n'
printf '核心报告: outputs_week4/week4_analysis.md\n'
printf '推荐查看: outputs_week4/auc_comparison.png\n'
printf '          outputs_week4/eer_comparison.png\n'
printf '          outputs_week4/improvement_heatmap.png\n'
