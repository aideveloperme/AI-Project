# DGX Spark is aarch64 (Grace CPU) + Blackwell GB10 GPU (sm_121). NGC PyTorch images are
# multi-arch and ship CUDA, cuDNN, TensorRT, Triton, Nsight Systems/Compute CLIs.
# Use a release that lists DGX Spark / GB10 support (25.10 or newer).
ARG NGC_TAG=25.10
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}-py3

WORKDIR /workspace/spark-edge-profiling
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# sanity: fail the build early if the GPU stack is incomplete
RUN python -c "import tensorrt, torch, torchvision, triton; print('TensorRT', tensorrt.__version__, 'torch', torch.__version__)"

COPY . .
RUN pip install --no-cache-dir --no-deps -e .
CMD ["bash"]
