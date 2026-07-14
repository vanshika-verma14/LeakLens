# LeakLens — Progress Tracker

*The durable, session-surviving source of truth. Update this every session — it's the memory that survives `/clear`. Claude Code should read this at the start of each session and tick boxes as work completes. Do NOT keep task state only in chat.*

---

## Current status
- **Phase:** Slice 1 spine complete (T1.1–T1.6). End-to-end pipeline runs.
- **Last done:** T1.5/T1.6 — `embed_corpus.py` builds a 240-vector Chroma store; `run_inversion_demo.py` samples (stratified, `--limit`), inverts, scores, prints per-category recall + threshold-free distribution. Verified on a 4-row subset (plain 1.00/pii 0.50/cred 0.50/struct 0.33).
- **Next action:** (1) user runs `python scripts/run_inversion_demo.py --limit 30` to eyeball per-category shape, then `--full` for the real distribution → set leak threshold. (2) optional corpus curation pass (T1.1). Then Slice 2 (defenses/study) or Slice 3 (tool wrapper).

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
- [~] T1.1 `corpus/corpus.jsonl` GENERATED (240 rows: 60 plain/pii/credential/structured) via `corpus/build_corpus.py`; validator + `tests/test_corpus.py` green. ← **awaiting your curation pass** (design + convention approved; label+value entities, reserved-range fakes)
- [x] T1.2 `leaklens/inversion/inverter.py` (Inverter: lazy load-once, CPU, encode/invert/roundtrip; `get_inverter()` singleton) + `tests/test_inverter.py`. Golden test is HONEST: low-entropy sentence, asserts ≥0.6 of key entities recovered (NOT exact-match), num_steps=5, `@slow` (run `pytest --runslow`). Calibration: 3/3 entities recovered at 5 steps, ~95s load+invert. Slow tests auto-set HF offline (see conftest).
- [x] T1.3 `adapters/base.py` (Sample dataclass + VectorStoreAdapter ABC: add/sample/count) + `chroma_adapter.py` + `tests/test_chroma_adapter.py` (5 tests). sample() returns Sample{id,vector,text,type,key_entities}; key_entities JSON-serialized in Chroma metadata (Chroma forbids list values). chromadb==1.5.9 installed (pins held). ARCHITECTURE.md adapter snippet updated to Sample shape.
- [x] T1.4 `leaklens/inversion/metrics.py` + `tests/test_metrics.py` (13 tests). Primary = key-entity recall, "found" = **case-insensitive substring** (your call; exact/fuzzy also computed into per-row `details`). Secondary = cosine (context only; stays 0.88–0.99 even when secret lost → not the verdict). `per_category_recall` + threshold-free `recall_distribution`. **No leak threshold hardcoded** — you set it from the distribution. Documented limitation: common-word labels (Order/Invoice/Transaction) can ci-match coincidentally → label recall overstates; value recovery is the honest signal (tests guard this).
- [x] T1.5 `scripts/embed_corpus.py` — encodes corpus.jsonl (GTR) → Chroma store at `results/corpus_store` (gitignored); idempotent rebuild. Verified: count 240.
- [x] T1.6 `scripts/run_inversion_demo.py` — spine: store → stratified sample → invert (batched) → `score_row` → print original↔recovered + per-category recall + threshold-free distribution; writes `results/inversion_demo.json`. **`--limit N` (default 30), `--full` opt-in**, `--num-steps`/`--seed`. Stratified subset via `stratified_sample` in adapters/base.py (+ `tests/test_sampling.py`, 4 tests). Verified with `--limit 4 --num-steps 5`: plain 1.00 / pii 0.50 / cred 0.50 / struct 0.33 — expected shape.
- **DoD:** ✅ demo prints original↔recovered + score; metric defensible (key-entity ci recall; DECISIONS D5/#6). Awaiting your `--limit 30` run to set the leak threshold from the real distribution.

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
- 2026-07-13 — T1.2: same cert-store issue also breaks `huggingface_hub` metadata calls (it phones home even for cached models). Models ARE cached (vec2text in HF hub cache, gtr-t5-base in torch sentence-transformers cache). Fix: run offline — `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. conftest sets these automatically when `--runslow` is passed (must be set before huggingface_hub import → argv peek at conftest top).
- _(add new learnings here so they survive context clears)_
