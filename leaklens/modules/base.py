"""The `Module` contract: one shape in, one shape out, and never crash the runner.

Every audit surface (inversion, cache_poison, ...) subclasses this so the runner and
report layer never special-case a module. The isolation rule is the whole point: a
`run` MUST catch its own errors and return an `INCONCLUSIVE` Finding rather than raise.
The runner (T3.5) wraps each `run` in a try/except as a second line of defence, but a
module that self-protects gives a *useful* INCONCLUSIVE (its own message) instead of a
generic "module crashed".
"""
from abc import ABC, abstractmethod

from leaklens.finding import Finding


class Module(ABC):
    """One audit surface. `target` is a built adapter; `cfg` is the loaded `Config`."""

    name: str

    @abstractmethod
    def run(self, target, cfg) -> Finding:
        """Run the audit and return a Finding (including INCONCLUSIVE). Must not raise."""
