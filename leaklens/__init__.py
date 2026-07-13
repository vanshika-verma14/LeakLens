"""LeakLens — a defensive self-audit tool for RAG infrastructure leakage.

Audits the layer beneath RAG apps for embedding-inversion and semantic-cache
leakage. See docs/ARCHITECTURE.md for the design; every scan module returns a
Finding.
"""

__version__ = "0.0.1"
