"""The one shape every store adapter speaks: a `Sample`, and the `VectorStoreAdapter`
contract that yields them.

`Sample` carries more than a bare (vector, text) pair on purpose: recovery is scored
*per category* in T1.4 and the Phase-2 study, so each sampled vector must remember its
`type` and `key_entities`. Real breach targets won't have those (text/type None,
key_entities empty) — that's expected, not an error.
"""
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
