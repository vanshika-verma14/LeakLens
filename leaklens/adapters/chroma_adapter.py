"""Chroma implementation of `VectorStoreAdapter` (the primary store, DECISIONS D4).

Chroma metadata values must be scalars — a list is rejected — so `key_entities` is
stored as a JSON string and parsed back on read. That serialization is the only subtle
part; everything else is a thin pass-through to a persistent Chroma collection.
"""
import json
import random

import chromadb
from chromadb.config import Settings

from leaklens.adapters.base import Sample, VectorStoreAdapter


class ChromaAdapter(VectorStoreAdapter):
    """Persist and sample vectors (with ground-truth metadata) in a Chroma collection."""

    def __init__(self, path, collection: str, *, space: str = "cosine"):
        client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False))
        self._col = client.get_or_create_collection(
            collection, metadata={"hnsw:space": space})

    def add(self, samples: list[Sample]) -> None:
        if not samples:
            return
        docs = [s.text for s in samples]
        kwargs = dict(
            ids=[s.id for s in samples],
            embeddings=[[float(x) for x in s.vector] for s in samples],
            # Chroma forbids list metadata -> serialize key_entities; type kept as str.
            metadatas=[{"type": s.type or "", "key_entities": json.dumps(s.key_entities)}
                       for s in samples],
        )
        if any(d is not None for d in docs):
            kwargs["documents"] = docs
        self._col.add(**kwargs)

    def count(self) -> int:
        return self._col.count()

    def sample(self, n: int, *, seed: int | None = None) -> list[Sample]:
        ids = self._col.get()["ids"]  # lightweight: ids only, no embeddings pulled
        rng = random.Random(seed)
        chosen = ids if n >= len(ids) else rng.sample(ids, n)
        if not chosen:
            return []
        got = self._col.get(ids=chosen, include=["embeddings", "documents", "metadatas"])
        return [self._to_sample(i, v, d, m) for i, v, d, m in
                zip(got["ids"], got["embeddings"], got["documents"], got["metadatas"])]

    @staticmethod
    def _to_sample(id_, vec, doc, meta) -> Sample:
        meta = meta or {}
        return Sample(
            id=id_,
            vector=[float(x) for x in vec],
            text=doc,
            type=(meta.get("type") or None),
            key_entities=json.loads(meta.get("key_entities", "[]")),
        )
