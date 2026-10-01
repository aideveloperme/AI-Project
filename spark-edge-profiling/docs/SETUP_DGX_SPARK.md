# Setting up the DGX Spark for profiling

## 1. What you are measuring

| Item | DGX Spark (GB10) | Why it matters here |
|---|---|---|
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) | aarch64: use NGC/arm64 wheels, never x86 PyPI CUDA wheels |
| GPU | Blackwell, 5th-gen tensor cores (FP4/FP8/INT8/FP16/BF16/TF32) | INT8 & FP8 paths via TensorRT |
| Memory | 128 GB LPDDR5x **unified** (CPU+GPU), ~273 GB/s | one DRAM pool → low bandwidth vs datacenter HBM, so many layers are memory-bound |
| Power | small-form-factor system power budget | report W and mJ/inference, watch thermals |

Consequences of unified memory:
* `nvidia-smi` shows `Memory-Usage: Not Supported` — normal. This project measures memory
  with the PyTorch allocator (`max_memory_allocated`) and TensorRT (`device_memory_size`).
* The OS page cache competes with the GPU for the same DRAM; drop it before long runs.
* Host→device copies are cheap but not free; TensorRT timings here exclude them (input already resident).

## 2. Host preparation (once)

```bash
# DGX OS ships the driver, Docker and the NVIDIA container toolkit
nvidia-smi                                  # GPU visible, driver loaded
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.10-py3 nvidia-smi   # container sees GPU
# (log in to nvcr.io first if pulls are refused: docker login nvcr.io, user $oauthtoken, NGC API key)
```

Use an NGC PyTorch tag that lists DGX Spark / GB10 support (25.10 or newer); set it with
`NGC_TAG=xx.yy ./docker/run.sh`.

### GPU performance counters (needed only for step 8, Nsight Compute)

Nsight Compute needs access to the counters that report DRAM bytes. Either run as root in
the container with `--cap-add SYS_ADMIN` (what `docker/run.sh` does) or allow all users:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
sudo update-initramfs -u && sudo reboot
```

Symptom if missing: `ERR_NVGPUCTRPERM` from `ncu`.

## 3. Measurement hygiene (do this before every full run)

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'   # free unified memory held by page cache
sudo cpupower frequency-set -g performance 2>/dev/null || true
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE > results/gpu_state_before.txt
```

* Close other GPU work (desktop sessions, notebooks, Ollama/LLM servers): they share the same DRAM bandwidth.
* Let the box idle ~1 min between power measurements; record ambient conditions.
* Clock locking (`nvidia-smi -lgc`) may be unsupported on GB10 — if it fails, rely on
  warm-up + many iterations + p50/p99, which every script reports.
* Repeat a configuration 3× if two variants differ by < 5 %; treat smaller deltas as noise.

## 4. Inside the container

```bash
./docker/run.sh
python scripts/00_env_check.py      # all lines should say [ok]
pytest -q tests                     # CPU correctness tests
./run_all.sh --quick                # smoke run
```

`00_env_check.py` reports missing pieces:

| Missing | Fix |
|---|---|
| `modelopt` | `pip install 'nvidia-modelopt[torch]'` (in the Dockerfile already) |
| `power` | `pip install nvidia-ml-py`; NVML must expose power for the GB10 |
| `ncu` | Nsight Compute CLI ships in the NGC image; on the host install `nsight-compute` |
| `tensorrt` | use the NGC PyTorch image; do not `pip install tensorrt` on top of it |

## 5. Useful extra tools

```bash
# timeline of one TensorRT inference (kernels, reformat layers, gaps)
nsys profile -o reports/trt_fp16 --trace cuda,nvtx \
  trtexec --loadEngine=engines/resnet50/fp16.engine --shapes=input:8x3x224x224 --iterations=100
# full Nsight Compute report for one layer, open in the GUI on a laptop
ncu --set full -o reports/conv2 --profile-from-start off \
  python -m sparkprof.ncu_target --arch resnet50 --layer layer3.0.conv2 --channels-last
# TensorRT's own per-layer profile of an engine built by this project
trtexec --loadEngine=engines/resnet50/int8_ptq.engine --shapes=input:8x3x224x224 \
  --dumpProfile --separateProfileRun --dumpLayerInfo
```
