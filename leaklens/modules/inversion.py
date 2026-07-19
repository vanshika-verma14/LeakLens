"""Surface 1: embedding-inversion leakage, wrapped behind the `Module` contract.

Reuses the proven Slice-1/2 core unchanged — `inverter` (load-once vec2text), `metrics`
(key-entity recall, DECISIONS D5), and `stratified_sample` — and rolls their per-row
scores into one `Finding`. This is the same pipeline as `scripts/run_inversion_demo.py`,
just assembled into a verdict instead of printed.

Two honesty rules live here:

* **No inverter, no score.** A public inverter exists only for GTR (DECISIONS D3). If the
  store's encoder isn't one we can invert, we return INCONCLUSIVE — "invertible in
  principle, no inverter available" — *before* loading any model, and never a made-up
  number (CLAUDE.md Tier 1).

* **The leak threshold is the human's, from config.** Verdict is `mean_recall >=
  cfg.inversion.recovery_threshold`; this module never hardcodes that line. The severity
  bands below are a separate reporting rollup, not the leak decision.
"""
from leaklens.adapters.base import stratified_sample
from leaklens.finding import Finding, Severity, Verdict
from leaklens.inversion import inverter as inverter_mod
from leaklens.inversion import metrics
from leaklens.modules.base import Module

OWASP = "LLM08: Vector and Embedding Weaknesses"
BATCH_SIZE = 8
MAX_EVIDENCE = 5


def _severity_for(score: float, verdict: Verdict) -> Severity:
    """Map a leakage score to a severity band (a reporting rollup, not the leak line).

    Distinct from the config recovery_threshold, which decides LEAK vs NO_LEAK. A cleared
    store caps at LOW so severity never contradicts the verdict.
    """
    if verdict is not Verdict.LEAK:
        return Severity.LOW
    if score >= 0.8:
        return Severity.CRITICAL
    if score >= 0.6:
        return Severity.HIGH
    if score >= 0.4:
        return Severity.MEDIUM
    return Severity.LOW


def _truncate(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


class InversionModule(Module):
    """Audit a vector store for embedding-inversion leakage → a `Finding`."""

    name = "inversion"

    def __init__(self, inverter=None):
        # Injectable for tests; defaults to the process-wide cached ~2GB inverter lazily,
        # so the runner constructs this module with no arguments and pays load cost once.
        self._inverter = inverter

    def _inconclusive(self, summary: str, *, remediation: str = "",
                      details: dict | None = None) -> Finding:
        return Finding(module=self.name, verdict=Verdict.INCONCLUSIVE, score=0.0,
                       severity=Severity.LOW, owasp=OWASP, summary=summary,
                       remediation=remediation, details=details or {})

    def run(self, target, cfg) -> Finding:
        try:
            return self._run(target, cfg)
        except Exception as exc:  # isolation: a module never crashes the runner
            return self._inconclusive(
                f"inversion module errored: {type(exc).__name__}: {exc}")

    def _run(self, target, cfg) -> Finding:
        encoder = cfg.vector_store.encoder
        # 1) Honest-failure gate FIRST — pure string check, no 2GB model load.
        if encoder != inverter_mod.ENCODER:
            return self._inconclusive(
                f"invertible in principle, no inverter available for '{encoder}'",
                remediation=(f"A public vec2text inverter is available for "
                             f"'{inverter_mod.ENCODER}'. Point LeakLens at a store built "
                             f"with that encoder to measure inversion leakage."),
                details={"encoder": encoder, "supported_encoder": inverter_mod.ENCODER})

        # 2) Sample — balanced across categories, reproducible (same as the demo).
        total = target.count()
        if total == 0:
            return self._inconclusive("no vectors to sample from the store",
                                      details={"encoder": encoder, "count": 0})
        all_samples = target.sample(total, seed=cfg.options.seed)
        rows = stratified_sample(all_samples, cfg.inversion.sample_size,
                                 seed=cfg.options.seed)
        if not rows:
            return self._inconclusive("no vectors to sample from the store",
                                      details={"encoder": encoder, "count": total})

        # 3) Invert + score, batched.
        inv = self._inverter or inverter_mod.get_inverter()
        import torch  # local: only needed once real models are in play
        scored = []  # (RecoveryScore, original_text, recovered_text)
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            emb = torch.tensor([s.vector for s in batch])
            recovered = inv.invert(emb, num_steps=cfg.inversion.num_steps)
            rec_emb = inv.encode(recovered)
            for i, s in enumerate(batch):
                row = {"id": s.id, "type": s.type, "key_entities": s.key_entities}
                score = metrics.score_row(row, recovered[i], mode="ci",
                                          orig_emb=s.vector, rec_emb=rec_emb[i])
                scored.append((score, s.text, recovered[i]))

        # 4) Aggregate.
        recovery_scores = [s for s, _, _ in scored]
        recalls = [s.recall for s in recovery_scores]
        mean_recall = sum(recalls) / len(recalls)
        per_category = metrics.per_category_recall(recovery_scores)
        distribution = metrics.recall_distribution(recovery_scores)

        # 5) Verdict from the human-set config threshold — never hardcoded here.
        threshold = cfg.inversion.recovery_threshold
        verdict = Verdict.LEAK if mean_recall >= threshold else Verdict.NO_LEAK
        severity = _severity_for(mean_recall, verdict)

        # 6) Evidence: the most-recovered rows, original <-> recovered.
        top = sorted(scored, key=lambda t: t[0].recall, reverse=True)[:MAX_EVIDENCE]
        evidence = [{"id": sc.id, "type": sc.type, "recall": round(sc.recall, 3),
                     "original": _truncate(orig), "recovered": _truncate(rec)}
                    for sc, orig, rec in top]

        if verdict is Verdict.LEAK:
            summary = (f"embedding inversion recovered key entities at mean recall "
                       f"{mean_recall:.2f} (>= threshold {threshold:.2f}) over "
                       f"{len(rows)} sampled vectors")
            remediation = (
                "Stored embeddings are reconstructible. Harden the store: add Gaussian "
                "noise to vectors before persisting — the defense sweep (studies/) shows "
                "recovery collapsing at small sigma while retrieval@k largely holds; pick "
                "the sigma from that privacy/utility curve. Treat the vector store as "
                "sensitive plaintext for encryption and incident scoping.")
        else:
            summary = (f"embedding inversion mean recall {mean_recall:.2f} below threshold "
                       f"{threshold:.2f} over {len(rows)} sampled vectors")
            remediation = ""

        return Finding(
            module=self.name, verdict=verdict, score=round(mean_recall, 4),
            severity=severity, owasp=OWASP, summary=summary, evidence=evidence,
            remediation=remediation,
            details={"encoder": encoder, "n_sampled": len(rows),
                     "mean_recall": mean_recall, "recovery_threshold": threshold,
                     "num_steps": cfg.inversion.num_steps,
                     "per_category_recall": per_category, "distribution": distribution})
