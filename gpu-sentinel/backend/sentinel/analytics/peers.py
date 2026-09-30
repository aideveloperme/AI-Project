"""GPU Peer Benchmarking.

Every GPU (and node) is compared against *comparable* peers: same GPU model,
same server type and same workload class by default (configurable). Robust
statistics (median / MAD) are used so one broken device cannot drag the
baseline it is being compared against.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sentinel.analytics.history import MetricHistory
from sentinel.telemetry.catalog import GPU_METRICS, NODE_METRICS, MetricSpec
from sentinel.telemetry.models import FleetSnapshot, NodeSample

MAD_SCALE = 1.4826  # MAD → σ for normally distributed data


@dataclass
class PeerStats:
    n: int
    median: float
    mean: float
    p10: float
    p90: float
    mad: float
    std: float
    min: float
    max: float

    @classmethod
    def of(cls, values: np.ndarray) -> "PeerStats":
        med = float(np.median(values))
        return cls(n=int(values.size), median=med, mean=float(values.mean()), p10=float(np.percentile(values, 10)),
                   p90=float(np.percentile(values, 90)), mad=float(np.median(np.abs(values - med))),
                   std=float(values.std()), min=float(values.min()), max=float(values.max()))


@dataclass
class PeerComparison:
    entity: str
    level: str
    metric: str
    label: str
    unit: str
    value: float
    peer: PeerStats
    deviation_pct: float
    robust_z: float
    percentile_rank: float
    historical_baseline: float | None
    historical_deviation_pct: float | None
    group: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["peer"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in d["peer"].items()}
        for k in ("value", "deviation_pct", "robust_z", "percentile_rank", "historical_baseline", "historical_deviation_pct"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        return d


def workload_kind(snap: FleetSnapshot, node: NodeSample) -> str:
    w = snap.workload(node.workload_id)
    return w.kind if w else "unknown"


def group_key(snap: FleetSnapshot, node: NodeSample, keys: list[str]) -> tuple[str, ...]:
    vals = {
        "gpu_model": node.gpu_model,
        "server_type": node.server_type,
        "cluster": node.cluster,
        "workload_kind": workload_kind(snap, node),
        "workload": node.workload_id or "none",
    }
    return tuple(vals[k] for k in keys)


def peer_groups(snap: FleetSnapshot, keys: list[str], min_group: int = 3) -> dict[str, list[str]]:
    """Assign each node to a peer group; fall back to coarser keys for tiny groups."""
    assignment: dict[str, tuple[str, ...]] = {}
    for depth in range(len(keys), 0, -1):
        ks = keys[:depth]
        groups: dict[tuple[str, ...], list[str]] = {}
        for n in snap.nodes:
            if n.node in assignment:
                continue
            groups.setdefault(group_key(snap, n, ks), []).append(n.node)
        for gk, members in groups.items():
            if len(members) >= min_group or depth == 1:
                for m in members:
                    assignment[m] = gk
    out: dict[str, list[str]] = {}
    for node, gk in assignment.items():
        out.setdefault(" | ".join(gk), []).append(node)
    return out


def _scale(stats: PeerStats, spec: MetricSpec) -> float:
    # Robust spread with floors: absolute (metric-specific) and relative (1% of median).
    return max(MAD_SCALE * stats.mad, spec.peer_abs_floor, abs(stats.median) * 0.01, 1e-9)


def compare(value: float, peers: np.ndarray, spec: MetricSpec, entity: str, group: str,
            history: MetricHistory | None = None, baseline_skip: int = 12) -> PeerComparison:
    stats = PeerStats.of(peers)
    dev = (value - stats.median) / abs(stats.median) * 100 if stats.median else 0.0
    z = (value - stats.median) / _scale(stats, spec)
    rank = float((peers < value).mean() * 100 + (peers == value).mean() * 50)
    hb = hd = None
    if history is not None:
        v = history.values(entity, spec.name)
        if v.size > baseline_skip + 10:
            hb = float(np.median(v[:-baseline_skip]))
            hd = (value - hb) / abs(hb) * 100 if hb else None
    return PeerComparison(entity=entity, level=spec.level, metric=spec.name, label=spec.label, unit=spec.unit,
                          value=value, peer=stats, deviation_pct=dev, robust_z=z, percentile_rank=rank,
                          historical_baseline=hb, historical_deviation_pct=hd, group=group)


class PeerBenchmark:
    """Computes peer comparisons for every GPU and node in a snapshot.

    Large groups (n ≥ ``exact_loo_below``) use full-group robust statistics,
    vectorised; small groups use exact leave-one-out so an outlier can't
    shift its own baseline. Historical baselines are only computed when
    ``with_history_baseline`` is set (API/incident paths, not the hot loop).
    """

    def __init__(self, group_keys: list[str] | None = None, min_group: int = 3, exact_loo_below: int = 12):
        self.group_keys = group_keys or ["gpu_model", "server_type", "workload_kind"]
        self.min_group = min_group
        self.exact_loo_below = exact_loo_below

    def _compare_group(self, keys: list[str], arr: np.ndarray, spec: MetricSpec, group: str,
                       history: MetricHistory | None, with_hb: bool, out: dict) -> None:
        n = arr.size
        if n <= self.exact_loo_below:
            for i, k in enumerate(keys):
                out.setdefault(k, {})[spec.name] = compare(float(arr[i]), np.delete(arr, i), spec, k, group,
                                                           history if with_hb else None)
            return
        stats = PeerStats.of(arr)
        scale = _scale(stats, spec)
        med = stats.median
        devs = (arr - med) / abs(med) * 100 if med else np.zeros(n)
        zs = (arr - med) / scale
        srt = np.sort(arr)
        ranks = (np.searchsorted(srt, arr, "left") + np.searchsorted(srt, arr, "right")) / 2 / n * 100
        for i, k in enumerate(keys):
            hb = hd = None
            if with_hb and history is not None:
                v = history.values(k, spec.name)
                if v.size > 22:
                    hb = float(np.median(v[:-12]))
                    hd = (arr[i] - hb) / abs(hb) * 100 if hb else None
            out.setdefault(k, {})[spec.name] = PeerComparison(
                entity=k, level=spec.level, metric=spec.name, label=spec.label, unit=spec.unit, value=float(arr[i]),
                peer=stats, deviation_pct=float(devs[i]), robust_z=float(zs[i]), percentile_rank=float(ranks[i]),
                historical_baseline=hb, historical_deviation_pct=hd, group=group)

    def run(self, snap: FleetSnapshot, history: MetricHistory | None = None, ma_window: int = 3,
            with_history_baseline: bool = False, only_nodes: set[str] | None = None) -> dict[str, dict[str, PeerComparison]]:
        """Returns {entity_key: {metric: PeerComparison}}.

        Values are compared using a short moving average (if history is
        available) to suppress single-sample noise.
        """
        groups = peer_groups(snap, self.group_keys, self.min_group)
        out: dict[str, dict[str, PeerComparison]] = {}
        for gname, members in groups.items():
            if only_nodes is not None and not (only_nodes & set(members)):
                continue
            nodes = [n for n in (snap.node(m) for m in members) if n is not None]
            gpus = [g for n in nodes for g in n.gpus]

            def val(entity: str, metric: str, raw: float) -> float:
                if history is None:
                    return raw
                ma = history.moving_average(entity, metric, ma_window)
                return raw if ma is None else ma

            for spec in GPU_METRICS:
                if not spec.peer_compare:
                    continue
                vals = {g.key: val(g.key, spec.name, g.metrics[spec.name]) for g in gpus if spec.name in g.metrics}
                if len(vals) < self.min_group:
                    continue
                self._compare_group(list(vals), np.fromiter(vals.values(), float, len(vals)), spec, gname,
                                    history, with_history_baseline, out)
            for spec in NODE_METRICS:
                if not spec.peer_compare:
                    continue
                vals = {n.node: val(n.node, spec.name, n.metrics[spec.name]) for n in nodes if spec.name in n.metrics}
                if len(vals) < self.min_group:
                    continue
                self._compare_group(list(vals), np.fromiter(vals.values(), float, len(vals)), spec, gname,
                                    history, with_history_baseline, out)
        return out
