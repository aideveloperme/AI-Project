#!/usr/bin/env bash
# One-time setup on the DGX Spark (GB10). Safe to re-run.
#   VLLM_IMAGE   container image (default vllm/vllm-openai:v0.31.0; an NGC
#                nvcr.io/nvidia/vllm:<yy.mm>-py3 image built for Spark also works)
#   HF_HOME      model cache (default ~/.cache/huggingface)
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODELS=(Qwen/Qwen3-8B Qwen/Qwen3-8B-FP8)

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Host"
uname -m
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || { echo "nvidia-smi failed"; exit 1; }
free -g | head -2

say "Docker + NVIDIA runtime"
docker info --format '{{json .Runtimes}}' | grep -q nvidia || echo "WARN: nvidia runtime not listed; '--gpus all' may still work via CDI"

say "Pull $VLLM_IMAGE"
docker pull "$VLLM_IMAGE"

say "Container sees the GPU (expect compute capability 12.x for GB10)"
docker run --rm --gpus all --entrypoint python3 "$VLLM_IMAGE" -c \
  "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda); \
print('device', torch.cuda.get_device_name(0), 'cc', torch.cuda.get_device_capability(0))" \
  || { echo "ERROR: image cannot use the GPU. Try an NGC vLLM image for DGX Spark: VLLM_IMAGE=nvcr.io/nvidia/vllm:<tag>"; exit 1; }

say "Download models into $HF_HOME (so server startup time excludes downloads)"
mkdir -p "$HF_HOME"
for m in "${MODELS[@]}"; do
  docker run --rm -v "$HF_HOME:/root/.cache/huggingface" ${HF_TOKEN:+-e HF_TOKEN} --entrypoint hf "$VLLM_IMAGE" \
    download "$m" --exclude "*.pth" "original/*" \
  || docker run --rm -v "$HF_HOME:/root/.cache/huggingface" ${HF_TOKEN:+-e HF_TOKEN} --entrypoint huggingface-cli \
    "$VLLM_IMAGE" download "$m"
done

say "Python harness"
python3 -m pip install -q -e ".[dev]"
python3 -m servebench kv-math --model Qwen/Qwen3-8B
echo; echo "Setup OK. Next: make gb10"
