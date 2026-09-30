# Running GPU Sentinel on a single DGX Spark

DGX Spark ships with DGX OS (Ubuntu, **arm64**), Docker and the NVIDIA Container Toolkit.
All GPU Sentinel images build or pull for arm64.

## Option A — simulated 12-node cluster (full product, demo)
```bash
git clone -b claude/gpu-sentinel-ai-mvp-kng07y https://github.com/aideveloperme/AI-Project.git
cd AI-Project/gpu-sentinel
docker compose up -d --build
# browse http://<spark-ip>:8080   admin / sentinel-admin
```

Optional local LLM (runs on the Spark GPU). The `ollama` service is behind the `llm` profile, so
start it explicitly, pull a model, then restart the backend: the backend picks its LLM at startup.
```bash
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
docker compose restart backend
docker compose logs backend | grep "GPU Sentinel"      # should show llm=ollama:llama3.1:...
```
(With `docker-compose.spark.yml`, add `-f docker-compose.spark.yml` to every command above.)

## Option B — monitor the Spark's own GPU
```bash
docker compose -f docker-compose.spark.yml up -d --build
docker compose -f docker-compose.spark.yml logs -f dcgm-exporter   # check it started
curl -s localhost:9100/metrics | head                              # node_exporter
```
Wait about 10 minutes for baselines to build (historical baseline: 60 samples), then run a real workload.

**Single-GPU limits.** Peer benchmarking needs at least 3 comparable GPUs, so one Spark gets **no** peer comparison
and no "X% below peers" headline. Detection relies on static thresholds, rolling z-score, the GPU's own
historical baseline and rate of change. Throughput, straggler and NCCL diagnosis also need the
`sentinel_workload_*` / `sentinel_nccl_*` metrics from your training code (not exported by default).

**GB10 caveats (unverified on hardware).** The GPU uses unified LPDDR5x memory, so DCGM may not report the
framebuffer, memory temperature or ECC fields. If dcgm-exporter exits on an unsupported field, delete that
line from `deploy/spark/dcgm-counters.csv` and restart it.

## Troubleshooting

* **`Bind for :::8080 failed: port is already allocated`**: the demo stack is still running.
  Run `docker compose down` (demo) first, or start the Spark stack on another port:
  `SENTINEL_PORT=8081 docker compose -f docker-compose.spark.yml up -d`.
* **`Not collecting DCP metrics`** in the dcgm-exporter log: profiling metrics (SM activity,
  tensor activity, DRAM bandwidth, PCIe bytes) aren't available. GPU Sentinel still works from
  utilization, clocks, temperature, power, throttle reasons, XID and host metrics; those panels show "—".
* Check that the GPU is being scraped:
  `docker compose -f docker-compose.spark.yml exec prometheus wget -qO- 'http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_TEMP'`

## Logging in

The Spark stack runs with demo mode off, so only the `admin` user exists (no `operator`/`viewer`).
Its password is `SENTINEL_ADMIN_PASSWORD` from `.env` if you created one (the example file sets
`change-me-too`), otherwise `sentinel-admin`. It is only applied when the database is first created.

Reset it, or add users, from the server:
```bash
docker compose -f docker-compose.spark.yml exec backend python -m sentinel.manage reset-password admin
docker compose -f docker-compose.spark.yml exec backend python -m sentinel.manage create-user alice operator
docker compose -f docker-compose.spark.yml exec backend python -m sentinel.manage list-users
```
