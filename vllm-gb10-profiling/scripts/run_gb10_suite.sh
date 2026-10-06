#!/usr/bin/env bash
# Run every experiment on the DGX Spark and build the before/after report.
# Usage: scripts/run_gb10_suite.sh [results_dir] [extra servebench args...]
# Takes ~2-3 h for the full suite; add --quick for a ~20 min smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/gb10}"; shift || true

mkdir -p "$OUT"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; uname -a
  nvidia-smi || true
  nvidia-smi -q -d CLOCK,POWER,TEMPERATURE 2>/dev/null | head -80 || true
  docker image inspect "${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}" --format '{{.Id}} {{.RepoDigests}}' || true
} > "$OUT/host_info.txt" 2>&1

# Stable numbers: nothing else should be using the GPU / unified memory.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "WARN: other GPU processes are running; results will be noisy:"; nvidia-smi --query-compute-apps=pid,name --format=csv
fi

python3 -m servebench run configs/experiments/0*.yaml --launcher docker --out "$OUT" --baseline 00_baseline "$@"
echo "Report: $OUT/REPORT.md"
