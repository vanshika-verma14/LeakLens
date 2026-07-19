"""Runner isolation — a crashing module becomes INCONCLUSIVE and never aborts the rest.

This is the guarantee ARCHITECTURE puts on the runner (FR-2/FR-5). Real modules already
self-isolate, so we prove the runner's own try/except with a module that deliberately
raises.
"""
from leaklens.config import Config, InversionConfig, Options, VectorStoreConfig
from leaklens.finding import Finding, Severity, Verdict
from leaklens.modules.base import Module
from leaklens.runner import run_modules


class BoomModule(Module):
    name = "boom"

    def run(self, target, cfg):
        raise RuntimeError("kaboom")


class GoodModule(Module):
    name = "good"

    def run(self, target, cfg):
        return Finding(module="good", verdict=Verdict.LEAK, score=1.0,
                       severity=Severity.HIGH, owasp="LLM08", summary="all recovered")


def make_cfg(modules):
    return Config(
        vector_store=VectorStoreConfig(type="chroma", path="./s", collection="docs"),
        modules=modules,
        options=Options(i_own_this_target=True),
        inversion=InversionConfig(recovery_threshold=0.6),
    )


REGISTRY = {"boom": BoomModule, "good": GoodModule}


def test_crashing_module_becomes_inconclusive_and_others_survive():
    findings = run_modules(object(), make_cfg(["boom", "good"]), registry=REGISTRY)

    assert [f.module for f in findings] == ["boom", "good"]  # order preserved, both present
    boom, good = findings
    assert boom.verdict is Verdict.INCONCLUSIVE
    assert "crashed" in boom.summary
    assert good.verdict is Verdict.LEAK        # the crash did not abort the good module
    assert good.score == 1.0


def test_good_module_alone_runs():
    findings = run_modules(object(), make_cfg(["good"]), registry=REGISTRY)
    assert len(findings) == 1
    assert findings[0].verdict is Verdict.LEAK


def test_unregistered_module_is_inconclusive_not_implemented():
    findings = run_modules(object(), make_cfg(["cache_poison"]), registry=REGISTRY)
    assert findings[0].verdict is Verdict.INCONCLUSIVE
    assert "not implemented" in findings[0].summary


def test_crash_first_still_runs_later_modules():
    # ordering guard: a crash as the FIRST module must not skip the ones after it.
    findings = run_modules(object(), make_cfg(["boom", "good", "boom"]), registry=REGISTRY)
    assert [f.verdict for f in findings] == [
        Verdict.INCONCLUSIVE, Verdict.LEAK, Verdict.INCONCLUSIVE]
