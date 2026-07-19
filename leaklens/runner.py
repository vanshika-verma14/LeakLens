"""Orchestrate the enabled modules and collect their Findings.

The load-bearing property is **isolation**: each module runs inside its own try/except, so
a module that raises becomes an INCONCLUSIVE Finding and the scan carries on with the rest
(FR-2/FR-5). Modules already self-isolate (they catch their own errors), but the runner is
the guarantee — a module that regresses and throws still can't abort the others.
"""
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.finding import Finding, Severity, Verdict
from leaklens.modules.inversion import InversionModule

# module name -> Module class. Only modules that exist; unbuilt names are handled honestly
# by run_modules (an INCONCLUSIVE "not implemented" Finding), never faked.
DEFAULT_MODULES = {
    "inversion": InversionModule,
}


def build_target(cfg):
    """Build the vector-store adapter named by the config. Reads only; never writes."""
    kind = cfg.vector_store.type
    if kind == "chroma":
        return ChromaAdapter(cfg.vector_store.path, cfg.vector_store.collection)
    if kind == "faiss":
        raise NotImplementedError(
            "FAISS adapter not built yet (Slice 5) — use a Chroma store for now")
    raise NotImplementedError(f"no adapter for vector_store.type '{kind}'")


def _inconclusive(module: str, summary: str) -> Finding:
    return Finding(module=module, verdict=Verdict.INCONCLUSIVE, score=0.0,
                   severity=Severity.LOW, owasp="", summary=summary)


def run_modules(target, cfg, *, registry=DEFAULT_MODULES) -> list[Finding]:
    """Run each enabled module in isolation, returning one Finding per module (in order)."""
    findings: list[Finding] = []
    for name in cfg.modules:
        module_cls = registry.get(name)
        if module_cls is None:
            findings.append(_inconclusive(name, f"module '{name}' not implemented"))
            continue
        try:
            findings.append(module_cls().run(target, cfg))
        except Exception as exc:  # the isolation guarantee — one crash never aborts the scan
            findings.append(_inconclusive(
                name, f"module '{name}' crashed: {type(exc).__name__}: {exc}"))
    return findings


def run_scan(cfg, *, registry=DEFAULT_MODULES) -> list[Finding]:
    """Build the target from the config and run every enabled module against it."""
    return run_modules(build_target(cfg), cfg, registry=registry)
