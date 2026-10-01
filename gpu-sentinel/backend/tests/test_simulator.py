from prometheus_client.parser import text_string_to_metric_families

from sentinel.simulator.cluster import ClusterSimulator
from sentinel.simulator.exporter import render_metrics
from sentinel.simulator.faults import Fault, FaultType


def test_topology_and_workloads():
    sim = ClusterSimulator(n_nodes=12, gpus_per_node=8, seed=1)
    snap = sim.tick(5)
    assert len(snap.nodes) == 12 and len(snap.all_gpus()) == 96
    assert {w.scheduler for w in snap.workloads} == {"slurm", "kubernetes"}
    assert snap.workload("78421").nodes[0] == "gpu-01"
    g = snap.nodes[0].gpus[0]
    for k in ("gpu_util", "sm_clock_mhz", "temp_c", "power_w", "hbm_bw_gbps", "nvlink_bw_gbps", "ecc_sbe_rate", "perf_index"):
        assert k in g.metrics
    for k in ("cpu_util", "ib_rx_gbps", "nccl_busbw_gbps", "throughput", "dataloader_wait_pct"):
        assert k in snap.nodes[0].metrics


def test_thermal_fault_ramps_temperature_and_throttles():
    sim = ClusterSimulator(seed=2)
    sim.warmup(60, 10)
    before = sim.tick(10).node("gpu-04").gpus[3].metrics
    sim.inject(Fault(FaultType.THERMAL, "gpu-04", gpu=3, severity=1.0))
    first = sim.tick(10).node("gpu-04").gpus[3].metrics
    for _ in range(20):
        snap = sim.tick(10)
    after = snap.node("gpu-04").gpus[3]
    assert first["temp_c"] < after.metrics["temp_c"]           # thermal inertia: it ramps
    assert after.metrics["temp_c"] > before["temp_c"] + 15
    assert after.metrics["sm_clock_mhz"] < before["sm_clock_mhz"] * 0.85
    assert "hw_thermal_slowdown" in after.throttle_reasons
    assert snap.node("gpu-04").metrics["throughput"] < snap.node("gpu-03").metrics["throughput"] * 0.9
    # other GPUs unaffected
    assert abs(snap.node("gpu-04").gpus[0].metrics["temp_c"] - before["temp_c"]) < 6


def test_clear_resets_fault_and_ecc_state():
    sim = ClusterSimulator(seed=3)
    f = sim.inject(Fault(FaultType.ECC_ERRORS, "gpu-02", gpu=1, severity=1.0))
    for _ in range(60):
        sim.tick(10)
    assert sim.nodes[1].gpus[1].counters["ecc_sbe"] > 100
    assert sim.clear(fault_id=f.id) == 1
    assert sim.nodes[1].gpus[1].last_xid == 0
    assert sim.active_faults() == []


def test_unknown_node_rejected():
    import pytest
    with pytest.raises(ValueError):
        ClusterSimulator(seed=1).inject(Fault(FaultType.THERMAL, "nope"))


def test_exporter_is_valid_prometheus_with_dcgm_names():
    sim = ClusterSimulator(seed=4)
    sim.tick(5)
    fams = {f.name: f for f in text_string_to_metric_families(render_metrics(sim))}
    for name in ("DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_SM_CLOCK", "DCGM_FI_PROF_DRAM_ACTIVE", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
                 "node_cpu_seconds", "node_infiniband_port_data_received_bytes", "sentinel_workload_throughput",
                 "sentinel_nccl_allreduce_busbw_gbps", "sentinel_node_info"):
        assert name in fams, name
    temp = fams["DCGM_FI_DEV_GPU_TEMP"].samples
    assert len(temp) == 96
    assert {"gpu", "UUID", "modelName", "Hostname", "node"} <= set(temp[0].labels)
