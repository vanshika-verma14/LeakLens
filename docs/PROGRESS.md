# LeakLens — Progress Tracker

*The durable, session-surviving source of truth. Update this every session — it's the memory that survives `/clear`. Claude Code should read this at the start of each session and tick boxes as work completes. Do NOT keep task state only in chat.*

---

## Current status
- **Phase:** Slice 0 complete → starting Slice 1 (inversion pipeline).
- **Last done:** Slice 0 scaffolding written + verified — `leaklens` imports (v0.0.1), `_compat` resource stub lets `import vec2text` succeed on Windows, `pytest` collects `conftest.py` cleanly (0 tests).
- **Next action:** Slice 1 task T1.1 — curate `corpus/corpus.jsonl` (~200–300 labelled sentences).

---

## Phase 0 — Smoke test + environment ✅ DONE
- [x] Python 3.11 venv created
- [x] `torch` + `vec2text` + `sentence-transformers` installed
- [x] Pinned `transformers==4.30.2`, `accelerate==0.21.0`
- [x] Windows `resource`-module stub working
- [x] Round-trip confirmed on CPU; behavior recorded (verbatim low-entropy, gist high-entropy)

## Slice 0 — Scaffolding
- [x] `requirements.txt` (pinned to working venv)
- [x] `.gitignore`, `pyproject.toml`, `leaklens/__init__.py`
- [x] `leaklens/_compat.py` (resource stub + force-CPU) — verified `import vec2text` works after it
- [x] `tests/__init__.py` + `tests/conftest.py` (tiny corpus + temp store-dir fixtures)
- [x] `git init` (already done); README stub present (empty, filled Phase 6)
- [ ] first commit (deferred — commit on user request)
- **DoD:** ✅ `pytest` runs (collects conftest, 0 tests; exit 5 = "no tests", expected), `python -c "import leaklens"` works. Commit pending user go-ahead.

## Slice 1 — Inversion pipeline (Phase 1)
- [ ] T1.1 `corpus/corpus.jsonl` curated (~200–300 labelled sentences) ← **your judgment**
- [ ] T1.2 `inverter.py` + `test_inverter.py` (golden round-trip)
- [ ] T1.3 `adapters/base.py` + `chroma_adapter.py` + test
- [ ] T1.4 `metrics.py` + `test_metrics.py` ← **you define + defend the threshold**
- [ ] T1.5 `scripts/embed_corpus.py` (corpus → Chroma)
- [ ] T1.6 `scripts/run_inversion_demo.py` (the spine, end-to-end)
- **DoD:** demo prints original↔recovered + score; metric defensible in two sentences.

## Slice 2 — Defense tradeoff study (Phase 2) — the finding
- [ ] T2.1 `defenses.py` (+σ noise, pooling, quantization) + test
- [ ] T2.2 `retrieval.py` (recall@k) + test
- [ ] T2.3 `studies/defense_sweep.py` (leakage + recall@k per level) ← **you design**
- [ ] T2.4 `studies/plot_tradeoff.py` → `results/tradeoff.png` (run on Colab)
- [ ] T2.5 Write one-paragraph finding in your own words
- **DoD:** tradeoff curve saved; crossover (or its absence) stated honestly.

## Slice 3 — Tool wrapper (Phase 3)
- [ ] T3.1 `finding.py` + test
- [ ] T3.2 `config.py` + `config.example.yaml` + ownership gate + test
- [ ] T3.3 `modules/base.py` + `modules/inversion.py` (+ honest-failure) + test
- [ ] T3.4 `report/owasp.py` + `scorecard.py` + `json_report.py` + `html_report.py`
- [ ] T3.5 `runner.py` (isolation) + `cli.py` + `__main__.py` + test
- **DoD:** `python -m leaklens scan --config config.yaml` runs end-to-end; terminal + HTML/JSON reports written.

## Slice 4 — Validation harness (Phase 3.5) — the moat
- [ ] T4.1 `scripts/build_validation_targets.py` (≥3 leaky + ≥3 hardened) ← **you design**
- [ ] T4.2 `validation/harness.py` + `test_validation.py`
- [ ] T4.3 `results/validation_matrix.md` generated
- **DoD:** tool flags all leaky, clears all hardened, reproducibly (fixed seed).

## Slice 5 — Surface 2: cache poisoning (Phase 4) — build only if 1–4 solid
- [ ] T5.1 `scripts/setup_gptcache.py`
- [ ] T5.2 `adapters/gptcache_adapter.py` (+ `faiss_adapter.py`) + tests
- [ ] T5.3 `modules/cache_poison.py` (canary, poison radius) ← **you design** + test
- **DoD:** `--module cache-poison` reports poison radius; loose flagged, tight cleared.

## Slice 6 — Stretch: Ghost Vectors (Phase 5) — cut freely
- [ ] `modules/ghost_vectors.py` + test
- **DoD:** shows deleted-vector recovery, or honestly reports the store hard-deletes.

## Slice 7 — Polish (Phase 6)
- [ ] Final `README.md` (leads with infra-layer gap + validation table + demo GIF)
- [ ] `docs/mechanisms.md` (attacks in your own words) ← **your whiteboard defense**
- [ ] `results/demo.gif`
- **DoD:** a stranger can read the README, run one command, and get a trustworthy report.

---

## Decisions / learnings log (append as you go)
- 2025 — Phase 0: smoke test green; metric = key-entity + similarity (not exact-match), evidence: high-entropy secrets mangle while names/structure survive.
- 2026-07-13 — Slice 0: pip in this venv hits `SSL: CERTIFICATE_VERIFY_FAILED` against PyPI (local trust-store issue). Workaround for future installs: `pip install <pkg> --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org`.
- _(add new learnings here so they survive context clears)_
