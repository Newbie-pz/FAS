#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=============================================='
printf '%s\n' '第三周追加域泛化方法'
printf '%s\n' '1) MixStyle'
printf '%s\n' '2) Fourier amplitude augmentation'
printf '%s\n\n' '=============================================='

bash scripts/run_week3_mixstyle.sh
bash scripts/run_week3_fourier.sh

printf '\n两组追加域泛化实验全部完成。\n'
