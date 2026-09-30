"""Deterministic / statistical anomaly detectors.

Implemented methods:
  1. static thresholds (evaluated on a moving average → no single-sample flapping)
  2. moving average smoothing (shared by all detectors)
  3. rolling standard deviation / z-score (short-window temporal)
  4. peer-group comparison (robust z-score vs. median/MAD of comparable peers)
  5. historical baseline comparison (long-window per-entity median)
  6. rate-of-change detection
Multi-metric correlation is done downstream in :mod:`sentinel.rca`.

Adding an ML detector (Isolation Forest, autoencoder, forecasting residuals…)
means implementing :class:`Detector` and registering it in
:func:`default_detectors`; its signals flow through RCA/incidents unchanged.
"""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from sentinel.analytics.history import MetricHistory
from sentinel.analytics.peers import PeerComparison
from sentinel.telemetry.catalog import CATALOG, MetricSpec
from sentinel.telemetry.models import FleetSnapshot

Severity = Literal["info", "warning", "critical"]
SEV_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class AnomalySignal:
    entity: str            # "gpu-04" or "gpu-04/gpu3"
    node: str
    gpu_index: int | None
    level: str             # gpu | node
    metric: str
    label: str
    unit: str
    category: str
    value: float
    expected: float | None
    deviation_pct: float | None
    zscore: float | None
    direction: Literal["high", "low"]
    severity: Severity
    method: str
    methods: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("value", "expected", "deviation_pct", "zscore"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


@dataclass
class DetectionContext:
    snapshot: FleetSnapshot
    history: MetricHistory
    peers: dict[str, dict[str, PeerComparison]]
    ma_window: int = 3


def _entity_meta(entity: str) -> tuple[str, int | None]:
    if "/gpu" in entity:
        node, g = entity.split("/gpu")
        return node, int(g)
    return entity, None


def _iter_values(ctx: DetectionContext):
    for n in ctx.snapshot.nodes:
        for k, v in n.metrics.items():
            s = CATALOG.get(("node", k))
            if s:
                yield n.node, s, v
        for g in n.gpus:
            for k, v in g.metrics.items():
                s = CATALOG.get(("gpu", k))
                if s:
                    yield g.key, s, v


def _bad(spec: MetricSpec, direction: str) -> bool:
    return spec.direction == "both" or spec.direction == f"{direction}_bad"


def _signal(entity: str, spec: MetricSpec, value: float, expected: float | None, dev: float | None,
            z: float | None, direction: str, severity: Severity, method: str, description: str) -> AnomalySignal:
    node, gpu = _entity_meta(entity)
    return AnomalySignal(entity=entity, node=node, gpu_index=gpu, level=spec.level, metric=spec.name,
                         label=spec.label, unit=spec.unit, category=spec.category, value=value, expected=expected,
                         deviation_pct=dev, zscore=z, direction=direction, severity=severity,  # type: ignore[arg-type]
                         method=method, methods=[method], description=description)


class Detector(abc.ABC):
    name: str

    @abc.abstractmethod
    def detect(self, ctx: DetectionContext) -> list[AnomalySignal]: ...


class StaticThresholdDetector(Detector):
    name = "static_threshold"

    def detect(self, ctx: DetectionContext) -> list[AnomalySignal]:
        out = []
        for entity, spec, raw in _iter_values(ctx):
            if spec.nonzero_is_bad and raw > 0:
                sev: Severity = "critical" if spec.crit is not None and raw >= spec.crit else "warning"
                out.append(_signal(entity, spec, raw, 0.0, None, None, "high", sev, self.name,
                                   f"{spec.label} is {raw:g} (expected 0)"))
                continue
            if spec.warn is None and spec.crit is None:
                continue
            ma = ctx.history.moving_average(entity, spec.name, ctx.ma_window)
            v = raw if ma is None else ma
            if spec.crit is not None and v >= spec.crit:
                out.append(_signal(entity, spec, v, spec.crit, None, None, "high", "critical", self.name,
                                   f"{spec.label} {v:.1f}{spec.unit} ≥ critical threshold {spec.crit:g}{spec.unit}"))
            elif spec.warn is not None and v >= spec.warn:
                out.append(_signal(entity, spec, v, spec.warn, None, None, "high", "warning", self.name,
                                   f"{spec.label} {v:.1f}{spec.unit} ≥ warning threshold {spec.warn:g}{spec.unit}"))
        return out


class PeerDeviationDetector(Detector):
    name = "peer_deviation"

    def __init__(self, z_threshold: float = 3.5):
        self.z = z_threshold

    def detect(self, ctx: DetectionContext) -> list[AnomalySignal]:
        out = []
        for entity, metrics in ctx.peers.items():
            for name, pc in metrics.items():
                spec = CATALOG[(pc.level, name)]
                direction = "high" if pc.robust_z > 0 else "low"
                if not _bad(spec, direction):
                    continue
                rel = abs(pc.deviation_pct) / 100
                if abs(pc.robust_z) < self.z or rel < spec.peer_min_rel_dev:
                    continue
                sev: Severity = "critical" if rel >= 2 * spec.peer_min_rel_dev and abs(pc.robust_z) >= 2 * self.z else "warning"
                out.append(_signal(entity, spec, pc.value, pc.peer.median, pc.deviation_pct, pc.robust_z, direction,
                                   sev, self.name,
                                   f"{spec.label} {pc.value:.1f}{spec.unit} is {abs(pc.deviation_pct):.1f}% "
                                   f"{'above' if direction == 'high' else 'below'} peer median {pc.peer.median:.1f}{spec.unit} "
                                   f"(n={pc.peer.n}, robust z={pc.robust_z:+.1f})"))
        return out


class RollingZScoreDetector(Detector):
    """Short-window temporal detector: moving average vs. rolling mean/std."""

    name = "rolling_zscore"

    def __init__(self, window: int = 60, recent: int = 3, z_threshold: float = 4.0, min_samples: int = 20):
        self.window, self.recent, self.z, self.min_samples = window, recent, z_threshold, min_samples

    def detect(self, ctx: DetectionContext) -> list[AnomalySignal]:
        out = []
        for entity, spec, _ in _iter_values(ctx):
            if not spec.temporal:
                continue
            v = ctx.history.values(entity, spec.name)
            if v.size < self.min_samples + self.recent:
                continue
            base = v[-(self.window + self.recent):-self.recent]
            recent = float(v[-self.recent:].mean())
            mu, sd = float(base.mean()), float(base.std())
            sd = max(sd, spec.peer_abs_floor / 2, abs(mu) * 0.005, 1e-9)
            z = (recent - mu) / sd
            direction = "high" if z > 0 else "low"
            rel = abs(recent - mu) / abs(mu) if mu else 0.0
            if abs(z) >= self.z and rel >= spec.peer_min_rel_dev and _bad(spec, direction):
                out.append(_signal(entity, spec, recent, mu, (recent - mu) / abs(mu) * 100 if mu else None, z, direction,
                                   "warning", self.name,
                                   f"{spec.label} changed from rolling mean {mu:.1f} to {recent:.1f}{spec.unit} (z={z:+.1f})"))
        return out


class HistoricalBaselineDetector(Detector):
    """Long-window per-entity baseline (median), excluding the recent window."""

    name = "historical_baseline"

    def __init__(self, recent: int = 6, min_history: int = 60, factor: float = 1.2, refresh_every: int = 30):
        self.recent, self.min_history, self.factor = recent, min_history, factor
        # Long-window medians barely move; recompute every N samples per series.
        self.refresh_every = refresh_every
        self._cache: dict[tuple[str, str], tuple[int, float]] = {}
        self._tick = 0

    def detect(self, ctx: DetectionContext) -> list[AnomalySignal]:
        out = []
        self._tick += 1
        for entity, spec, _ in _iter_values(ctx):
            if not spec.temporal:
                continue
            v = ctx.history.values(entity, spec.name)
            if v.size < self.min_history:
                continue
            cached = self._cache.get((entity, spec.name))
            if cached is None or self._tick - cached[0] >= self.refresh_every:
                cached = (self._tick, float(np.median(v[:-self.recent])))
                self._cache[(entity, spec.name)] = cached
            base = cached[1]
            cur = float(v[-self.recent:].mean())
            if not base:
                continue
            dev = (cur - base) / abs(base)
            direction = "high" if dev > 0 else "low"
            if abs(dev) >= spec.peer_min_rel_dev * self.factor and abs(cur - base) > spec.peer_abs_floor and _bad(spec, direction):
                out.append(_signal(entity, spec, cur, base, dev * 100, None, direction, "warning", self.name,
                                   f"{spec.label} {cur:.1f}{spec.unit} vs. own historical baseline {base:.1f}{spec.unit} ({dev * 100:+.1f}%)"))
        return out


class RateOfChangeDetector(Detector):
    """Least-squares slope (per minute) over a *time* window (default 2 min, ≥1 min span),
    independent of the collection interval. Also requires the absolute change over the
    window to exceed the metric's noise floor, so sensor jitter never trips it."""

    name = "rate_of_change"
    LIMITS = {("gpu", "temp_c"): 4.0, ("gpu", "mem_temp_c"): 4.0, ("gpu", "power_w"): 200.0}
    REL_LIMITS = {("node", "throughput"): 0.15, ("node", "nccl_busbw_gbps"): 0.25}

    def __init__(self, window_s: float = 120.0, min_span_s: float = 60.0, min_samples: int = 5):
        self.window_s, self.min_span_s, self.min_samples = window_s, min_span_s, min_samples

    def detect(self, ctx: DetectionContext) -> list[AnomalySignal]:
        out = []
        for entity, spec, _ in _iter_values(ctx):
            key = (spec.level, spec.name)
            if key not in self.LIMITS and key not in self.REL_LIMITS:
                continue
            t = ctx.history.times(entity, spec.name)
            if t.size < self.min_samples:
                continue
            mask = t >= t[-1] - self.window_s
            x, y = t[mask], ctx.history.values(entity, spec.name)[mask]
            if x.size < self.min_samples or x[-1] - x[0] < self.min_span_s:
                continue
            slope = float(np.polyfit((x - x[0]) / 60.0, y, 1)[0])  # units per minute
            change = slope * (x[-1] - x[0]) / 60.0
            direction = "high" if slope > 0 else "low"
            if not _bad(spec, direction):
                continue
            if key in self.LIMITS and abs(slope) >= self.LIMITS[key] and abs(change) >= 2 * max(spec.peer_abs_floor, 1.0):
                out.append(_signal(entity, spec, float(y[-1]), float(y[0]), None, None, direction, "warning", self.name,
                                   f"{spec.label} changing at {slope:+.1f}{spec.unit}/min"))
            elif key in self.REL_LIMITS and y.mean() and abs(slope) / abs(y.mean()) >= self.REL_LIMITS[key] \
                    and abs(change) / abs(y.mean()) >= spec.peer_min_rel_dev:
                out.append(_signal(entity, spec, float(y[-1]), float(y[0]), slope / abs(y.mean()) * 100, None, direction,
                                   "warning", self.name, f"{spec.label} changing at {slope / abs(y.mean()) * 100:+.1f}%/min"))
        return out


def default_detectors() -> list[Detector]:
    return [StaticThresholdDetector(), PeerDeviationDetector(), RollingZScoreDetector(),
            HistoricalBaselineDetector(), RateOfChangeDetector()]


# Priority when merging several methods' signals for the same entity+metric.
METHOD_PRIORITY = ["static_threshold", "peer_deviation", "historical_baseline", "rolling_zscore", "rate_of_change"]


def merge_signals(signals: list[AnomalySignal]) -> list[AnomalySignal]:
    """One signal per (entity, metric, direction); keep the most informative, record all methods."""
    merged: dict[tuple[str, str, str], AnomalySignal] = {}
    for s in signals:
        k = (s.entity, s.metric, s.direction)
        cur = merged.get(k)
        if cur is None:
            merged[k] = s
            continue
        methods = sorted(set(cur.methods) | set(s.methods), key=lambda m: METHOD_PRIORITY.index(m) if m in METHOD_PRIORITY else 99)
        best = cur
        if SEV_ORDER[s.severity] > SEV_ORDER[cur.severity] or (
            SEV_ORDER[s.severity] == SEV_ORDER[cur.severity]
            and METHOD_PRIORITY.index(s.method) < METHOD_PRIORITY.index(cur.method)
        ):
            best = s
        best.methods = methods
        # Prefer peer-relative numbers when available (they're what operators compare).
        peer = next((x for x in (cur, s) if x.method == "peer_deviation"), None)
        if peer is not None and best is not peer:
            best.expected, best.deviation_pct, best.zscore = peer.expected, peer.deviation_pct, peer.zscore
        merged[k] = best
    return list(merged.values())


def run_detectors(ctx: DetectionContext, detectors: list[Detector] | None = None) -> list[AnomalySignal]:
    out: list[AnomalySignal] = []
    for d in detectors or default_detectors():
        out.extend(d.detect(ctx))
    return merge_signals(out)
