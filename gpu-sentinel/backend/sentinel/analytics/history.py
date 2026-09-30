"""Bounded in-memory metric history used by temporal detectors.

Long-term history lives in the database (TimescaleDB hypertables); this ring
buffer only keeps what the online detectors need (default ~2h at 10s), so
analysis never has to hit the database in the hot path.

Each series is a pre-allocated numpy buffer of 2×maxlen that is compacted
when full, so ``values()`` is an O(1) view and appends are amortised O(1).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from sentinel.telemetry.models import FleetSnapshot


class _Series:
    __slots__ = ("ts", "v", "pos", "maxlen")

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.ts = np.empty(2 * maxlen)
        self.v = np.empty(2 * maxlen)
        self.pos = 0

    def append(self, ts: float, value: float) -> None:
        if self.pos == 2 * self.maxlen:
            self.ts[: self.maxlen] = self.ts[self.maxlen:]
            self.v[: self.maxlen] = self.v[self.maxlen:]
            self.pos = self.maxlen
        self.ts[self.pos] = ts
        self.v[self.pos] = value
        self.pos += 1

    def window(self) -> slice:
        return slice(max(0, self.pos - self.maxlen), self.pos)


class MetricHistory:
    def __init__(self, maxlen: int = 720):
        self.maxlen = maxlen
        self._data: dict[tuple[str, str], _Series] = {}

    def add(self, entity: str, metric: str, ts: datetime, value: float) -> None:
        s = self._data.get((entity, metric))
        if s is None:
            s = self._data[(entity, metric)] = _Series(self.maxlen)
        s.append(ts.timestamp(), value)

    def ingest(self, snap: FleetSnapshot) -> None:
        for n in snap.nodes:
            for k, v in n.metrics.items():
                self.add(n.node, k, snap.timestamp, v)
            for g in n.gpus:
                for k, v in g.metrics.items():
                    self.add(g.key, k, snap.timestamp, v)

    def values(self, entity: str, metric: str) -> np.ndarray:
        s = self._data.get((entity, metric))
        if s is None:
            return np.empty(0)
        return s.v[s.window()]

    def times(self, entity: str, metric: str) -> np.ndarray:
        s = self._data.get((entity, metric))
        if s is None:
            return np.empty(0)
        return s.ts[s.window()]

    def series(self, entity: str, metric: str, last: int | None = None) -> list[tuple[datetime, float]]:
        t, v = self.times(entity, metric), self.values(entity, metric)
        if last:
            t, v = t[-last:], v[-last:]
        return [(datetime.fromtimestamp(a, tz=timezone.utc), float(b)) for a, b in zip(t, v)]

    def moving_average(self, entity: str, metric: str, window: int = 3) -> float | None:
        v = self.values(entity, metric)
        if v.size == 0:
            return None
        return float(v[-window:].mean())

    def __len__(self) -> int:
        return len(self._data)
