# GITEX demo script (≈ 8 minutes)

**Setup (before the booth opens)**

```bash
docker compose up -d --build          # add --profile llm for the local LLM
open http://localhost:8080            # admin / sentinel-admin
```

Let the stack run for at least 3 minutes before the first visitor. That gives peer baselines and
Prometheus `rate()` windows time to warm up. Keep the **Overview** page on the big screen.
Everything runs on the laptop: unplug the network cable to prove it.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Overview: 96 GPUs green, 0 incidents | "This is a 12-node H100 cluster: a Slurm training job and a Kubernetes inference service. The telemetry is the same DCGM and node_exporter data your cluster already produces." |
| 0:45 | **Demo Control** → *Thermal problem* → gpu-04, GPU 3, severity 0.9 → Inject | "Let's break the cooling on one GPU." |
| 1:00 | Node Details → gpu-04 → trend charts | Temperature ramps and SM clock falls. "No threshold has fired yet. We compare GPU 3 with 63 identical GPUs running the same job." |
| 1:30 | Incidents → INC-… | "One incident, not five alerts. Temperature, clock, tensor activity and throughput were correlated into a single thermal/power event." |
| 2:00 | Incident page | Read the summary aloud: *"gpu-04 is performing ~18% below its peer baseline… Possible contributing factor: thermal throttling (high confidence)"*. Point at the **OBSERVED / INFERRED / RECOMMENDED** columns, then at *"GPU utilization is normal"*. "It tells you what is **not** the problem too." |
| 3:00 | Peer comparison table | "Every number comes with its peer median and distribution. The AI never makes up a metric: a validator rejects any number that isn't in the evidence." |
| 3:30 | **Ask Sentinel**: "Why is GPU-04 slow?" → "Why did training job 78421 slow down?" → "Compare node 4 with healthy nodes." | "The copilot has no shell. It calls a fixed set of read-only tools over live telemetry. The tool it used is shown under each answer." |
| 5:00 | Demo Control → scenario **software-regression** | "Now the hard case: the job slows down, but every GPU, CPU and network metric is normal." → RCA page shows *Application / software-layer slowdown* at **medium** confidence. "It points you at the software layer, and it doesn't pretend to be certain." |
| 6:00 | Settings → *Offline / on-premise status* | "Telemetry never leaves the site. The LLM runs locally, cloud AI is off unless you switch it on, and every action is audited." |
| 7:00 | Demo Control → **Clear all** | Incidents auto-resolve after about a minute. Re-inject thermal: the new incident shows *Recurring*. |

**Backup plans**

* LLM slow or absent → explanations come from the deterministic engine (label: *deterministic (offline)*). The content is identical in structure.
* No Docker → `make dev` (backend with in-process simulator + Next dev server). Same demo, no Prometheus.
* Too many visitors → scenario **mixed-incidents** creates three different incidents at once: thermal, power cap and NCCL.

**Fault catalogue** (all available in Demo Control): thermal, power cap, HBM bandwidth degradation, clock
reduction, ECC errors (XID 48), NVLink degradation, PCIe downtraining, CPU bottleneck, host memory pressure,
InfiniBand degradation, NCCL communication bottleneck, software regression.
