"""Store adapters. Each hides a concrete vector store (Chroma, FAISS, ...) behind the
one `VectorStoreAdapter` interface, so the inversion module samples vectors without
knowing or caring which store they live in (see docs/ARCHITECTURE.md).
"""
