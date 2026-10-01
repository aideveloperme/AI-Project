#!/usr/bin/env bash
# Build (once) and start the profiling container on the DGX Spark.
#   ./docker/run.sh                 interactive shell
#   ./docker/run.sh ./run_all.sh    full pipeline
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-sparkprof:latest}
NGC_TAG=${NGC_TAG:-25.10}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build --build-arg NGC_TAG="$NGC_TAG" -t "$IMAGE" -f docker/Dockerfile .
fi

# --cap-add SYS_ADMIN : lets Nsight Compute read GPU performance counters (DRAM bytes)
# --ipc=host / ulimits : recommended by NVIDIA for PyTorch data loaders
# the project dir is bind-mounted so data, engines, results and reports persist on the host
exec docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add SYS_ADMIN \
  -v "$PWD":/workspace/spark-edge-profiling \
  -v "${HOME}/.cache/torch":/root/.cache/torch \
  -w /workspace/spark-edge-profiling \
  "$IMAGE" "${@:-bash}"
