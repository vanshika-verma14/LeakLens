# LeakLens — Progress Tracker

*The durable, session-surviving source of truth. Update this every session — it's the memory that survives `/clear`. Claude Code should read this at the start of each session and tick boxes as work completes. Do NOT keep task state only in chat.*

---

## Current status
- **Phase:** Slice 2 pipeline COMPLETE (T2.1–T2.4). Tradeoff study measures leakage vs utility across a σ grid and renders the curve; awaiting the real `--full`/num_steps=20 run for the citable numbers + the written finding (T2.5).
- **Last done:** calibrated the sweep (full-store recall@k utility over all 240, fine σ grid 0→0.5) so the tradeoff is visible; added recall@1 alongside recall@5; built `studies/plot_tradeoff.py` (PREVIEW-labelled tradeoff curve, comfortable-band shading via documented CLI thresholds). Preview (`--limit 12 --num-steps 5`): leakage → 0 by σ=0.03; utility@5 flat to ~0.05 then bends; utility@1 bends earlier (0.933 at σ=0.03) → no comfortable band at leakage≤0 ∧ utility≥0.99, the honest tension.
- **Next action:** run the real `--full --num-steps 20` sweep (Colab), re-render the plot, then write T2.5 (one-paragraph finding). Re-confirm the 0.6 leak threshold (D9) against that full run when convenient. Then Slice 3 (tool wrapper).

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
- [x] T1.1 `corpus/corpus.jsonl` — 240 curated rows (60 plain/pii/credential/structured) via `corpus/build_corpus.py`; validator + `tests/test_corpus.py` green. Curation pass done: frames capped ≤3, unique credential secret values, sentence-initial framing-words dropped as entities, label+value tagging, reserved-range fakes.
- [x] T1.2 `leaklens/inversion/inverter.py` (Inverter: lazy load-once, CPU, encode/invert/roundtrip; `get_inverter()` singleton) + `tests/test_inverter.py`. Golden test is HONEST: low-entropy sentence, asserts ≥0.6 of key entities recovered (NOT exact-match), num_steps=5, `@slow` (run `pytest --runslow`). Calibration: 3/3 entities recovered at 5 steps, ~95s load+invert. Slow tests auto-set HF offline (see conftest).
- [x] T1.3 `adapters/base.py` (Sample dataclass + VectorStoreAdapter ABC: add/sample/count) + `chroma_adapter.py` + `tests/test_chroma_adapter.py` (5 tests). sample() returns Sample{id,vector,text,type,key_entities}; key_entities JSON-serialized in Chroma metadata (Chroma forbids list values). chromadb==1.5.9 installed (pins held). ARCHITECTURE.md adapter snippet updated to Sample shape.
- [x] T1.4 `leaklens/inversion/metrics.py` + `tests/test_metrics.py` (13 tests). Primary = key-entity recall, "found" = **case-insensitive substring** (your call; exact/fuzzy also computed into per-row `details`). Secondary = cosine (context only; stays 0.88–0.99 even when secret lost → not the verdict). `per_category_recall` + threshold-free `recall_distribution`. **No leak threshold hardcoded** — you set it from the distribution. Documented limitation: common-word labels (Order/Invoice/Transaction) can ci-match coincidentally → label recall overstates; value recovery is the honest signal (tests guard this).
- [x] T1.5 `scripts/embed_corpus.py` — encodes corpus.jsonl (GTR) → Chroma store at `results/corpus_store` (gitignored); idempotent rebuild. Verified: count 240.
- [x] T1.6 `scripts/run_inversion_demo.py` — spine: store → stratified sample → invert (batched) → `score_row` → print original↔recovered + per-category recall + threshold-free distribution; writes `results/inversion_demo.json`. **`--limit N` (default 30), `--full` opt-in**, `--num-steps`/`--seed`. Stratified subset via `stratified_sample` in adapters/base.py (+ `tests/test_sampling.py`, 4 tests). Verified with `--limit 4 --num-steps 5`: plain 1.00 / pii 0.50 / cred 0.50 / struct 0.33 — expected shape.
- **DoD:** ✅ demo prints original↔recovered + score; metric defensible (key-entity ci recall; DECISIONS D5/#6). Awaiting your `--limit 30` run to set the leak threshold from the real distribution.

## Slice 2 — Defense tradeoff study (Phase 2) — the finding
- [x] T2.1 `leaklens/inversion/defenses.py` (`gaussian_noise` σ-sweep, `quantize`; dim-preserving so both leakage+utility measurable) + `tests/test_defenses.py` (5).
- [x] T2.2 `leaklens/inversion/retrieval.py` (`recall_at_k`, cosine, clean-query model) + `tests/test_retrieval.py` (5).
- [x] T2.3 `studies/defense_sweep.py` — per σ, measures BOTH leakage (invert defended → ci recall) AND utility (recall@1 + recall@k) on the SAME defended array; writes `results/sweep.json`. **CALIBRATED (commit bfbe55b):** utility now queries the FULL 240-vector defended store, not the `--limit` subset (a tiny subset self-retrieved at flat 1.0 and hid the drop); leakage still inverts only the subset, sliced from that same defended array (coherent point). Default σ grid widened/fined to `0,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5` to catch both the sub-0.1 leakage collapse and the higher-σ utility break. Added recall@1 (strict nearest-neighbour) beside recall@5 → both keys in the JSON. Preview (`--limit 12 --num-steps 5`): leakage 0.54→0.00 by σ=0.03; utility@5 flat ~1.0 until σ≈0.07; utility@1 bends earlier (0.933 at σ=0.03).
- [x] T2.4 `studies/plot_tradeoff.py` → `results/tradeoff.png`. Pure/fast (no model, Agg backend); auto-discovers every `utility_recall_at_N` key → plots leakage + @1 + @5; shades the comfortable band (leakage ≤ `--leak-max` ∧ utility ≥ `--util-min`, both documented CLI args, printed on the figure — no silent threshold) or an honest "no band" note; PREVIEW caption from JSON metadata (num_steps/n/N/seed). matplotlib==3.11.0 pinned in requirements.txt (pip check clean). With @1 included at util-min=0.99 the band vanishes — the honest privacy/utility tension.
- [ ] T2.5 Write one-paragraph finding in your own words (after the real `--full` curve)
- **DoD:** tradeoff curve saved; crossover (or its absence) stated honestly. ← pipeline + plot done; awaiting real `--full`/num_steps=20 run for citable numbers + T2.5.

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
- 2026-07-14 — T1.4/threshold: **leak threshold set to recall ≥ 0.6 = LEAK (provisional)** — DECISIONS D9. Calibrated from the `--limit 30` distribution: 0.6 separates high-leak prose (plain median 1.00) from resistant high-entropy secrets (credential/structured median ≤0.50); PII straddles (honest mixed case). From real data, not a priori. Re-confirm against the full 240-row run.
- 2026-07-14 — T2.3 calibration: the sweep's utility was reading a flat 1.0 because recall@k was measured on the `--limit` subset (a handful of vectors trivially retrieve themselves). Fix: measure utility against the FULL 240-vector defended store while leakage still inverts only the subset (sliced from the same defended array). Widened/fined the σ grid to `0→0.5`. Result: utility actually moves and a real curve appears (commit bfbe55b).
- 2026-07-14 — T2.4/recall@1: added recall@1 beside recall@5 (strict — nearest neighbour must be the target). @1 bends earlier (0.933 at σ=0.03 vs @5=1.0), so requiring leakage≤0 ∧ utility≥0.99 for BOTH leaves NO comfortable band — the honest privacy/utility tension the two-line view hid. Band thresholds are documented CLI args on `plot_tradeoff.py`, never silent.
- 2026-07-14 — matplotlib installed via the cert workaround (`--index-url https://pypi.org/simple --trusted-host …`) and pinned `matplotlib==3.11.0`; `pip check` clean, load-bearing pins (transformers 4.30.2 / accelerate 0.21.0 / numpy 1.26.4 / torch 2.13.0) intact. Plotter forces the `Agg` backend and keeps unicode out of stdout prints (Windows cp1252 can't encode σ/≤).
- _(add new learnings here so they survive context clears)_
