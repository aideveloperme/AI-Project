from collections import Counter

import numpy as np
import pytest

from sentinel.analytics.detectors import (AnomalySignal, DetectionContext, PeerDeviationDetector, RateOfChangeDetector,
                                          RollingZScoreDetector, StaticThresholdDetector, merge_signals)
from sentinel.analytics.history import MetricHistory
from sentinel.analytics.peers import PeerBenchmark, PeerStats, compare, peer_groups
from sentinel.simulator.faults import Fault, FaultType
from sentinel.telemetry.catalog import CATALOG
from tests.helpers import Pipeline


def test_peer_stats_robust_to_outlier():
    vals = np.array([100.0] * 9 + [10.0])
    st = PeerStats.of(vals)
    assert st.median == 100.0 and st.mean < 100
    pc = compare(10.0, vals[:-1], CATALOG[("gpu", "sm_clock_mhz")], "x/gpu0", "g")
    assert pc.deviation_pct == pytest.approx(-90.0)
    assert pc.robust_z < -3.5


def test_history_ring_buffer_bounded():
    from datetime import datetime, timezone
    h = MetricHistory(maxlen=10)
    for i in range(35):
        h.add("e", "m", datetime.fromtimestamp(i, tz=timezone.utc), float(i))
    v = h.values("e", "m")
    assert v.size == 10 and v[0] == 25 and v[-1] == 34
    assert h.moving_average("e", "m", 3) == pytest.approx(33.0)
    assert len(h.series("e", "m", last=4)) == 4


def test_peer_groups_split_by_workload_and_fallback():
    p = Pipeline(seed=5)
    p.step()
    groups = peer_groups(p.snap, ["gpu_model", "server_type", "workload_kind"], 3)
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [4, 8]  # training vs inference
    # A group key that isolates each node falls back to coarser keys
    groups = peer_groups(p.snap, ["gpu_model", "workload"], 20)
    assert sum(len(v) for v in groups.values()) == 12


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_no_false_positives_on_healthy_fleet(seed):
    p = Pipeline(seed=seed)
    counts = Counter()
    for _ in range(80):
        for s in p.step():
            counts[(s.entity, s.metric)] += 1
    assert sum(counts.values()) == 0, counts.most_common(5)


def test_static_threshold_uses_moving_average():
    p = Pipeline(seed=6)
    p.step(20)
    p.snap.node("gpu-01").gpus[0].metrics["temp_c"] = 95.0  # single spike, not in history MA
    ctx = DetectionContext(p.snap, p.hist, p.peers)
    assert not [s for s in StaticThresholdDetector().detect(ctx) if s.metric == "temp_c"]


def test_detectors_catch_thermal_fault():
    p = Pipeline(seed=7)
    p.step(70)
    p.sim.inject(Fault(FaultType.THERMAL, "gpu-03", gpu=2, severity=1.0))
    p.step(4)
    ctx = DetectionContext(p.snap, p.hist, p.peers)
    roc = RateOfChangeDetector().detect(ctx)
    assert any(s.entity == "gpu-03/gpu2" and s.metric == "temp_c" for s in roc)
    # rolling z-score is the onset detector: it fires while the change is fresh
    rz = RollingZScoreDetector().detect(ctx)
    assert any(s.entity == "gpu-03/gpu2" and s.metric == "temp_c" for s in rz)
    p.step(8)
    ctx = DetectionContext(p.snap, p.hist, p.peers)
    peer = PeerDeviationDetector().detect(ctx)
    hits = {(s.entity, s.metric) for s in peer}
    assert ("gpu-03/gpu2", "temp_c") in hits and ("gpu-03/gpu2", "sm_clock_mhz") in hits
    assert all(s.node == "gpu-03" for s in peer)
    static = StaticThresholdDetector().detect(ctx)
    assert any(s.metric == "temp_c" and s.entity == "gpu-03/gpu2" for s in static)  # ≥85°C (MA)


def test_merge_keeps_one_signal_with_all_methods():
    base = dict(entity="n/gpu0", node="n", gpu_index=0, level="gpu", metric="temp_c", label="t", unit="°C",
                category="thermal", value=90, direction="high", description="")
    a = AnomalySignal(**base, expected=85, deviation_pct=None, zscore=None, severity="warning", method="static_threshold",
                      methods=["static_threshold"])
    b = AnomalySignal(**base, expected=70, deviation_pct=28.0, zscore=9.0, severity="critical", method="peer_deviation",
                      methods=["peer_deviation"])
    m = merge_signals([a, b])
    assert len(m) == 1
    assert set(m[0].methods) == {"static_threshold", "peer_deviation"}
    assert m[0].severity == "critical" and m[0].deviation_pct == 28.0


def test_peer_benchmark_leave_one_out_small_groups():
    p = Pipeline(seed=8)
    p.step(3)
    pb = PeerBenchmark()
    res = pb.run(p.snap)
    node_pc = res["gpu-10"]["throughput"]
    assert node_pc.peer.n == 3  # 4 inference nodes, leave-one-out
    gpu_pc = res["gpu-01/gpu0"]["temp_c"]
    assert gpu_pc.peer.n == 64  # 8 training nodes × 8 GPUs, full-group stats
