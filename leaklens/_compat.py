"""Windows / CPU compatibility shims.

Import this FIRST in any entrypoint that touches vec2text:

    import leaklens._compat  # noqa: F401

Why: vec2text imports the Unix-only stdlib ``resource`` module at import time,
which does not exist on Windows, so ``import vec2text`` raises ModuleNotFoundError.
We register a no-op stub before that import happens. We also pin execution to CPU
— the target laptop's iGPU/NPU are unusable for torch (see docs/DECISIONS.md).
"""
import os
import sys
import types


def _install_resource_stub() -> None:
    """Register a no-op ``resource`` module so ``import vec2text`` works on Windows.

    On real Unix the module already exists and we leave it untouched.
    """
    if "resource" in sys.modules:
        return
    try:
        import resource  # noqa: F401  (present on Unix — nothing to stub)
        return
    except ModuleNotFoundError:
        pass
    stub = types.ModuleType("resource")
    stub.getrlimit = lambda *a, **k: (0, 0)
    stub.setrlimit = lambda *a, **k: None
    stub.RLIMIT_AS = stub.RLIMIT_DATA = stub.RLIM_INFINITY = 0
    sys.modules["resource"] = stub


def force_cpu() -> None:
    """Keep torch/transformers on CPU (CPU-only target — docs/DECISIONS.md)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# Importing this module is the whole point — apply the shims on import.
_install_resource_stub()
force_cpu()
