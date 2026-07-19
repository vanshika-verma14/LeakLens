"""The validation harness — the moat.

A reproducible known-safe / known-vulnerable matrix that proves LeakLens's verdicts
are trustworthy: it must flag the known-leaky stores (LEAK) and clear the known-hardened
ones (NO_LEAK), deterministically under a fixed seed. Building the targets lives in
`scripts/build_validation_targets.py`; running the tool over them and scoring the matrix
lives here in `harness.py`.
"""
