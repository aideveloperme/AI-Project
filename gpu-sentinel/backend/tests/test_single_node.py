"""Single GPU (e.g. DGX Spark): idle → training must not look like an anomaly,
but a real degradation while training must still be caught."""
import random
from datetime import timedelta

from sentinel.analytics.detectors import DetectionContext, run_detectors
from sentinel.analytics.history import MetricHistory
from sentinel.analytics.peers import PeerBenchmark
from sentinel.telemetry.models import FleetSnapshot, GPUSample, NodeSample, utcnow

rng = random.Random(1)


def snap(t, busy: bool, clock: float = 2400.0) -> FleetSnapshot:
    n = lambda v, s: v + rng.gauss(0, s)  # noqa: E731
    g = {"gpu_util": n(97, 1) if busy else n(1, 0.5), "temp_c": n(82, 0.4) if busy else n(46, 0.4),
         "power_w": n(77, 2) if busy else n(11, 0.5), "sm_clock_mhz": n(clock, 8) if busy else n(400, 5),
         "mem_clock_mhz": 4266.0}
    node = {"cpu_util": n(12, 1) if busy else n(4.6, 0.5), "mem_used_pct": n(24, 0.3) if busy else n(5.7, 0.2),
            "load1": n(3, 0.3) if busy else n(0.5, 0.1)}
    if busy:
        node["throughput"] = n(18000 * clock / 2400, 150)
    gpu = GPUSample(node="spark-01", index=0, uuid="GPU-x", model="NVIDIA GB10", metrics=g)
    return FleetSnapshot(timestamp=t, source="test", nodes=[
        NodeSample(node="spark-01", cluster="dgx-spark", server_type="DGX Spark", gpu_model="NVIDIA GB10",
                   metrics=node, gpus=[gpu])])


def run(phases):
    hist, pb, t, out = MetricHistory(), PeerBenchmark(), utcnow(), []
    for busy, cycles, clock in phases:
        for _ in range(cycles):
            t += timedelta(seconds=10)
            s = snap(t, busy, clock)
            hist.ingest(s)
            out.append(run_detectors(DetectionContext(s, hist, pb.run(s, hist))))
    return out


def test_idle_to_training_is_not_an_anomaly():
    results = run([(False, 90, 2400), (True, 40, 2400)])
    flagged = {(s.metric, s.method) for r in results[90:] for s in r}
    assert flagged == set(), flagged


def test_degradation_during_training_is_detected():
    # 90 idle, 80 cycles of steady training (learns the active baseline), then clocks drop 25%.
    results = run([(False, 90, 2400), (True, 80, 2400), (True, 12, 1800)])
    late = {(s.metric, s.method) for r in results[-4:] for s in r}
    assert ("sm_clock_mhz", "historical_baseline") in late or ("sm_clock_mhz", "rolling_zscore") in late, late
    assert any(m == "throughput" for m, _ in late), late
