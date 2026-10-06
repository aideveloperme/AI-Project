"""Deterministic synthetic workloads.

Each workload produces chat ``messages`` with controlled lengths. Text is
drawn from a list of common English words, which Qwen/Llama BPE tokenizers
encode at ~1 token per word, so ``input_tokens`` is an approximation; the
*actual* prompt tokens are always taken from the server's ``usage``.

Workload kinds
--------------
random         Unique prompts. Defeats the prefix cache - a pure prefill/decode test.
shared_prefix  N long shared "system prompts" (RAG context, tool specs) followed by a
               short unique question. This is what the prefix cache is built for.
long_context   Long unique documents + short output. Stresses KV capacity and
               ``--max-model-len``; this is where preemption appears.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

WORDS = (
    "the of and to in is was for on that with as by at from his it an were are which this be or "
    "had not but first one their its new after who they have her she two been other when there all "
    "during into school time may years more most only over city some world would where later up such "
    "used many can state about national out known university united then made system between under "
    "three also part each work well both people group life name since team season game early year "
    "number including river however found film music north second series several company because law "
    "until became against while war south served public same main area power four high history before "
    "under government around these days west support local member market small family water place "
    "data model memory cache token batch server latency kernel request block layer engine stream queue"
).split()


def make_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(max(1, n_words)))


@dataclass
class WorkloadSpec:
    name: str
    kind: str = "random"  # random | shared_prefix | long_context
    input_tokens: int = 512
    input_jitter: float = 0.0  # +/- fraction applied to the unique part
    output_tokens: int = 128
    shared_prefix_tokens: int = 0
    num_prefixes: int = 1
    seed: int = 1234
    prefix_seed: int | None = None  # shared-prefix text seed; defaults to ``seed``
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkloadSpec:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Prompt:
    messages: list[dict[str, Any]]
    max_tokens: int
    est_input_tokens: int


class Workload:
    """Infinite, reproducible prompt stream for a :class:`WorkloadSpec`."""

    def __init__(self, spec: WorkloadSpec):
        if spec.kind not in {"random", "shared_prefix", "long_context"}:
            raise ValueError(f"unknown workload kind {spec.kind!r}")
        self.spec = spec
        self._rng = random.Random(spec.seed)
        self._i = 0
        prefix_rng = random.Random((spec.seed if spec.prefix_seed is None else spec.prefix_seed) ^ 0x5EED)
        self._prefixes = (
            [
                f"[context {p}] " + make_text(prefix_rng, spec.shared_prefix_tokens)
                for p in range(max(1, spec.num_prefixes))
            ]
            if spec.kind == "shared_prefix"
            else []
        )

    def _unique_len(self) -> int:
        base = self.spec.input_tokens
        j = self.spec.input_jitter
        if j > 0:
            base = int(base * self._rng.uniform(1 - j, 1 + j))
        return max(8, base)

    def next(self) -> Prompt:
        s = self.spec
        self._i += 1
        n = self._unique_len()
        # A unique tag at the *start* of the unique part guarantees no accidental
        # prefix-cache hits between "unique" prompts.
        unique = f"req-{s.seed}-{self._i}: " + make_text(self._rng, n)
        if s.kind == "shared_prefix":
            prefix = self._prefixes[self._rng.randrange(len(self._prefixes))]
            messages = [
                {"role": "system", "content": prefix},
                {"role": "user", "content": unique + "\nSummarise the context above."},
            ]
            est = s.shared_prefix_tokens + n
        else:
            messages = [{"role": "user", "content": unique + "\nContinue this text."}]
            est = n
        return Prompt(messages=messages, max_tokens=s.output_tokens, est_input_tokens=est)


def for_level(spec: WorkloadSpec, level: float, salt: int = 0) -> WorkloadSpec:
    """Per-level variant: fresh unique prompts, same shared prefixes.

    Without this, every sweep level would replay identical prompts and the
    prefix cache would serve later levels from earlier ones - a classic
    benchmarking bug that inflates cache-on results.
    """
    prefix_seed = spec.seed if spec.prefix_seed is None else spec.prefix_seed
    return replace(spec, seed=spec.seed + 7919 * (int(level * 1000) + 1) + salt, prefix_seed=prefix_seed)
