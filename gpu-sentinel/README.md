# GPU Sentinel AI

**AI-powered performance and health intelligence for NVIDIA GPU clusters. Runs fully on-premise.**

GPU Sentinel turns raw DCGM, node_exporter, InfiniBand, NCCL and scheduler telemetry into
**Detect → Correlate → Diagnose → Explain → Recommend**:

> *gpu-04 is performing 18.4% below its peer baseline.*
> **Observed:** GPU 3 SM clock 1,416 MHz, 22.6% below peer median · GPU temperature 88.5 °C vs 67.0 °C · throttle reason `hw_thermal_slowdown` · GPU utilization **normal**.
> **Inferred:** thermal throttling (high confidence).
> **Recommended:** 1. Check rack cooling and airflow. 2. Compare thermal history with neighbouring GPUs. 3. Inspect throttle reasons with `nvidia-smi`. 4. Run `dcgmi diag -r 3` in a maintenance window.

Deterministic analytics do the detection: peer benchmarking, statistical detectors and a rule-based
RCA engine. The AI layer only **explains** that evidence, and a validator throws out any AI answer that
contains a number or a cause that isn't in the evidence.

* 📐 Architecture and all 15 design deliverables: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* 🎤 Trade-show demo script: [docs/DEMO.md](docs/DEMO.md)
* 🖥️ Single DGX Spark: [docs/DGX_SPARK.md](docs/DGX_SPARK.md)

## Quick start

### Full on-prem stack (Docker)

```bash
cd gpu-sentinel
cp .env.example .env              # change the secrets for anything beyond a demo
docker compose up -d --build      # simulator → Prometheus → TimescaleDB → backend → dashboard
open http://localhost:8080        # admin / sentinel-admin  (operator / viewer accounts in demo mode)

docker compose --profile llm up -d                                   # optional local LLM
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
```

### Without Docker (development)

```bash
make install        # Python venv + npm install
make backend        # API on :8000, in-process simulator, SQLite
make frontend       # dashboard on :3000 (proxies /api to :8000)
make test           # backend test suite
```

### Connect a real cluster

1. Run [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) and node_exporter on each GPU node
   (enable the `infiniband` and `pressure` collectors).
2. In Prometheus, add a `node` label to every series (relabel examples are in `deploy/prometheus/prometheus.yml`).
3. Configure the backend:
   `SENTINEL_TELEMETRY_SOURCE=prometheus SENTINEL_PROMETHEUS_URL=http://prometheus:9090 SENTINEL_DEMO_MODE=false`.
4. Optional: export `sentinel_workload_*` / `sentinel_nccl_*` metrics from your training framework to get
   throughput, straggler and NCCL diagnosis. The names are listed in `sentinel/simulator/exporter.py`.

## What's inside

| Component | Path | Notes |
|---|---|---|
| Telemetry simulator | `backend/sentinel/simulator/` | 12×8 H100, Slurm + K8s workloads, 12 injectable fault types, DCGM-compatible `/metrics` |
| Collectors | `backend/sentinel/telemetry/` | PromQL vendor mapping (NVIDIA today, AMD/Intel-ready), canonical model |
| Peer benchmarking | `backend/sentinel/analytics/peers.py` | median/MAD robust z, p10/p90, percentile, historical baseline |
| Detectors | `backend/sentinel/analytics/detectors.py` | static threshold (moving average), rolling z-score, peer deviation, historical baseline, rate of change |
| RCA engine | `backend/sentinel/rca/engine.py` | 12 correlation rules with supporting/contradicting evidence and confidence |
| Incidents | `backend/sentinel/incidents/service.py` | gating, dedup, escalation, recurrence, auto-resolve, workflow |
| AI | `backend/sentinel/ai/` | local LLM (Ollama / OpenAI-compatible), grounding validator, offline fallback |
| Ask Sentinel | `backend/sentinel/assistant/` | natural language → 9 allow-listed read-only tools |
| Security | `backend/sentinel/auth/`, `api/` | JWT, API keys, RBAC, audit log, encrypted credentials, rate limiting |
| Dashboard | `frontend/` | Next.js static export, 14 pages, dark/light theme, no CDN dependencies |
| Deploy | `docker-compose.yml`, `deploy/k8s/` | network isolation, non-root containers, NetworkPolicies |

## Verified behaviour

* **Zero false positives** on a healthy fleet over 80 cycles (3 seeds, 3 s and 10 s intervals). This is an automated test.
* **Correct top diagnosis for all 12 fault types**: a parametrised golden test.
* End to end through a **real Prometheus**: simulator exporter → Prometheus 2.53 → PromQL → backend. A thermal
  fault on gpu-04 GPU 3 produced incident *"Thermal throttling on gpu-04"* (high confidence, −18.4% vs peers)
  with no incidents anywhere else.
* One analysis cycle for 96 GPUs takes about 70 ms.

## Configuration

All settings are `SENTINEL_*` environment variables (`backend/sentinel/config.py`). The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `TELEMETRY_SOURCE` | `simulator` | `simulator` (in-process) or `prometheus` |
| `ANALYSIS_INTERVAL_S` | `10` | analysis cycle length |
| `PEER_GROUP_KEYS` | `gpu_model,server_type,workload_kind` | how peers are chosen (`cluster`, `workload` also available) |
| `INCIDENT_MIN_CYCLES` / `INCIDENT_RESOLVE_CYCLES` | `2` / `6` | noise control |
| `LLM_PROVIDER` | `auto` | `auto` (Ollama if reachable), `template`, `ollama`, `openai_compatible` |
| `ALLOW_CLOUD_LLM` | `false` | cloud LLM endpoints are refused unless true |
| `ALLOW_EXTERNAL_NOTIFICATIONS` | `false` | Slack/Teams/PagerDuty blocked unless true |
| `JWT_SECRET`, `SECRET_KEY`, `BOOTSTRAP_ADMIN_PASSWORD` | dev values | **must** be set in production |
| `RETENTION_SAMPLES_DAYS` / `_INCIDENTS_DAYS` / `_AUDIT_DAYS` | 7 / 365 / 365 | data retention |
