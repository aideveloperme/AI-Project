#!/usr/bin/env bash
# Nsight Systems capture of one experiment (complements the torch-profiler traces).
# The server runs with --profiler-config '{"profiler":"cuda"}' so /start_profile and
# /stop_profile call cudaProfilerStart/Stop, and nsys captures only that window.
#
# Usage: scripts/nsys_profile.sh configs/experiments/00_baseline.yaml [out_dir]
# Output: <out_dir>/<name>.nsys-rep  (open with Nsight Systems GUI, or `nsys stats`)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="$1"; NAME="$(basename "$CFG" .yaml)"; OUT="$(realpath -m "${2:-results/gb10/$NAME/nsys}")"
PORT="${PORT:-8000}"; VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$OUT"

# Render the experiment's vllm flags (minus any torch profiler config) via the harness.
mapfile -t SERVE < <(python3 - "$CFG" "$PORT" <<'PY'
import sys
from servebench.experiment import load_experiment
from servebench.launcher import build_command
exp = load_experiment(sys.argv[1]); exp["server"]["profiler-config"] = {"profiler": "cuda"}
cmd = build_command(exp, "local", int(sys.argv[2]), None)
print("\n".join(cmd))
PY
)
MODEL_NAME="$(python3 -c "from servebench.experiment import load_experiment as l; print(l('$CFG')['served_model_name'])")"

docker rm -f servebench-nsys >/dev/null 2>&1 || true
docker run -d --name servebench-nsys --gpus all --ipc=host -p "$PORT:$PORT" \
  -v "$HF_HOME:/root/.cache/huggingface" -v "$OUT:/out" --entrypoint bash "$VLLM_IMAGE" -c \
  "command -v nsys >/dev/null || { echo 'nsys not in image: use an NGC image or install nsight-systems-cli'; exit 3; }; \
   nsys profile -o /out/$NAME --force-overwrite true --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
     --capture-range=cudaProfilerApi --capture-range-end=stop $(printf '%q ' "${SERVE[@]}")"
trap 'docker stop -t 60 servebench-nsys >/dev/null 2>&1 || true' EXIT

echo "waiting for server..."; until curl -sf "localhost:$PORT/health" >/dev/null; do
  docker inspect -f '{{.State.Running}}' servebench-nsys | grep -q true || { docker logs servebench-nsys | tail -20; exit 1; }
  sleep 3; done

python3 -m servebench loadtest --model "$MODEL_NAME" --levels 1 --num-requests 2   # warm up (not captured)
curl -sf -X POST "localhost:$PORT/start_profile"
python3 -m servebench loadtest --model "$MODEL_NAME" --levels 8 --num-requests 16 --output-tokens 64
curl -sf -X POST "localhost:$PORT/stop_profile"
sleep 20  # nsys writes the report after the capture range ends
ls -la "$OUT"
echo "Summaries: nsys stats --report cuda_gpu_kern_sum $OUT/$NAME.nsys-rep"
