from datetime import timedelta

from sentinel.analytics.detectors import DetectionContext, run_detectors
from sentinel.analytics.history import MetricHistory
from sentinel.analytics.peers import PeerBenchmark
from sentinel.rca.engine import RCAEngine
from sentinel.simulator.cluster import ClusterSimulator
from sentinel.telemetry.models import utcnow


class Pipeline:
    """In-memory analytics pipeline over the simulator (no DB/API)."""

    def __init__(self, seed: int = 1, dt: float = 10.0, **sim_kwargs):
        self.sim = ClusterSimulator(seed=seed, **sim_kwargs)
        self.hist = MetricHistory()
        self.pb = PeerBenchmark()
        self.rca = RCAEngine()
        self.dt = dt
        self.t = utcnow()
        self.i = 0

    def step(self, n: int = 1):
        for _ in range(n):
            snap = self.sim.tick(self.dt)
            self.i += 1
            snap.timestamp = self.t + timedelta(seconds=self.dt * self.i)
            self.hist.ingest(snap)
            peers = self.pb.run(snap, self.hist)
            sig = run_detectors(DetectionContext(snap, self.hist, peers))
        self.snap, self.peers, self.signals = snap, peers, sig
        return sig

    def diagnose(self, node: str):
        ev = self.rca.build_evidence(node, self.snap, self.signals, self.peers)
        return ev, self.rca.evaluate(ev)
