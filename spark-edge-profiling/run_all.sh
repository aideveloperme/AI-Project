#!/usr/bin/env bash
# End-to-end pipeline. Pass --quick for a ~15-minute smoke run; extra args go to every step,
# e.g.  ./run_all.sh --set bench.iters=500
set -euo pipefail
cd "$(dirname "$0")"
ARGS=("$@")
step() { echo; echo "================ $1 ================"; shift; python "$@" "${ARGS[@]}"; }

step "0  environment"            scripts/00_env_check.py
step "1  roofline ceilings"      scripts/01_microbench_peaks.py
step "2  fine-tune baselines"    scripts/02_train_baselines.py
step "3  precision sweep"        scripts/03_precision_sweep.py
step "4  Conv+BN+ReLU fusion"    scripts/04_fusion_experiment.py
step "5  layout & bandwidth"     scripts/05_layout_bandwidth.py
step "6  model surgery"          scripts/06_model_surgery.py
step "7  roofline (analytic)"    scripts/07_roofline.py
if command -v ncu >/dev/null; then
  step "8  Nsight Compute DRAM"  scripts/08_ncu_layers.py
  step "7b roofline (+ncu bytes)" scripts/07_roofline.py
else
  echo "ncu not found: skipping measured DRAM traffic (roofline uses compulsory bytes)"
fi
step "9  report"                 scripts/09_make_report.py
echo; echo "Done -> reports/REPORT.md"
