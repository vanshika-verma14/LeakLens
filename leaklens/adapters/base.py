"""The one shape every store adapter speaks: a `Sample`, and the `VectorStoreAdapter`
contract that yields them.

`Sample` carries more than a bare (vector, text) pair on purpose: recovery is scored
*per category* in T1.4 and the Phase-2 study, so each sampled vector must remember its
`type` and `key_entities`. Real breach targets won't have those (text/type None,
key_entities empty) — that's expected, not an error.
"""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Sample:
    """One stored vector plus the ground-truth metadata needed to score its recovery."""

    id: str
    vector: list[float]
    text: str | None = None          # None for a real breach target; present for our corpus
    type: str | None = None          # plain | pii | credential | structured
    key_entities: list[str] = field(default_factory=list)


class VectorStoreAdapter(ABC):
    """Read/populate a vector store without exposing which store it is.

    Adapters do NOT embed text — they take precomputed vectors. Encoding stays in
    `leaklens.inversion.inverter` so the encoder is loaded once and in one place.
    """

    @abstractmethod
    def add(self, samples: list[Sample]) -> None:
        """Insert samples (id + vector + optional text/type/key_entities)."""

    @abstractmethod
    def sample(self, n: int, *, seed: int | None = None) -> list[Sample]:
        """Return up to `n` samples. `seed` makes the draw reproducible (NFR-3)."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently in the store."""


def stratified_sample(samples: list[Sample], limit: int | None, *,
                      seed: int | None = None) -> list[Sample]:
    """Return a subset of `samples` balanced across `Sample.type`, round-robin.

    A plain random draw of, say, 30 rows could land mostly in one category and hide the
    per-category shape we want to inspect; round-robin across the types keeps every
    category represented. Reproducible under `seed`. `limit` that is None, <= 0, or
    >= len(samples) returns all of them (order preserved).
    """
    if limit is None or limit <= 0 or limit >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    by_type: dict[str, list[Sample]] = {}
    for s in samples:
        by_type.setdefault(s.type or "unknown", []).append(s)
    for group in by_type.values():
        rng.shuffle(group)
    order = sorted(by_type)                 # deterministic category order
    cursors = {t: 0 for t in order}
    out: list[Sample] = []
    while len(out) < limit:
        progressed = False
        for t in order:
            if cursors[t] < len(by_type[t]):
                out.append(by_type[t][cursors[t]])
                cursors[t] += 1
                progressed = True
                if len(out) == limit:
                    break
        if not progressed:                  # every category exhausted
            break
    return out
