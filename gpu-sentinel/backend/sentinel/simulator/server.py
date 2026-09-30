"""Standalone simulator service (``python -m sentinel.simulator.server``).

Exposes ``/metrics`` for Prometheus to scrape and a small fault-injection API
used by the dashboard's demo controls. Runs as its own container in
docker-compose so it behaves exactly like an exporter on a real cluster.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from sentinel.simulator.cluster import ClusterSimulator
from sentinel.simulator.exporter import render_metrics
from sentinel.simulator.faults import FAULT_DESCRIPTIONS, Fault, FaultType

TICK_S = float(os.getenv("SIM_TICK_SECONDS", "5"))
sim = ClusterSimulator(
    n_nodes=int(os.getenv("SIM_NODES", "12")),
    gpus_per_node=int(os.getenv("SIM_GPUS_PER_NODE", "8")),
    cluster=os.getenv("SIM_CLUSTER", "dxb-ai-01"),
    random_faults=os.getenv("SIM_RANDOM_FAULTS", "false").lower() == "true",
    random_fault_rate_per_hour=float(os.getenv("SIM_RANDOM_FAULT_RATE", "2")),
)


class InjectRequest(BaseModel):
    type: FaultType
    node: str
    gpu: int | None = None
    severity: float = Field(0.8, ge=0.05, le=1.0)
    duration_s: float | None = Field(900, ge=10)


async def _loop() -> None:
    while True:
        sim.tick(TICK_S)
        await asyncio.sleep(TICK_S)


@asynccontextmanager
async def lifespan(_: FastAPI):
    sim.tick(TICK_S)
    task = asyncio.create_task(_loop())
    yield
    task.cancel()


app = FastAPI(title="GPU Sentinel Cluster Simulator", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "nodes": len(sim.nodes), "active_faults": len(sim.active_faults())}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return render_metrics(sim)


@app.get("/api/fault-types")
def fault_types() -> list[dict]:
    return [{"type": t.value, "description": d} for t, d in FAULT_DESCRIPTIONS.items()]


@app.get("/api/faults")
def list_faults() -> list[dict]:
    return [f.to_dict() for f in sim.active_faults()]


@app.post("/api/faults")
def inject(req: InjectRequest) -> dict:
    try:
        f = sim.inject(Fault(type=req.type, node=req.node, gpu=req.gpu, severity=req.severity, duration_s=req.duration_s))
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return f.to_dict()


@app.delete("/api/faults")
def clear_all() -> dict:
    return {"cleared": sim.clear()}


@app.delete("/api/faults/{fault_id}")
def clear(fault_id: str) -> dict:
    return {"cleared": sim.clear(fault_id=fault_id)}


@app.get("/api/nodes")
def nodes() -> list[dict]:
    return [{"node": n.name, "gpus": len(n.gpus), "workload": n.workload_id} for n in sim.nodes]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SIM_PORT", "9400")))
