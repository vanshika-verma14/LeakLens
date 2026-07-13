# LeakLens — Development Workflow (the "making" plan)

*How we actually build this with Claude Code without it becoming a vibe-coded mess. Read `BUILD_PLAN.md` for the phase roadmap and `ARCHITECTURE.md` for the design; this file is the **process** and the **complete file manifest in build order**.*

---

## Why this exists

The failure mode we're avoiding is well documented: unstructured generation works for throwaway demos but collapses on multi-file projects because **context degrades** and **ambiguity gets filled with the model's guesses**. The antidotes are simple and non-negotiable here: (1) a stable `CLAUDE.md`, (2) plan before editing, (3) durable state in `PROGRESS.md` not chat memory, (4) verify by running, (5) keep it simple.

---

## The core loop (repeat for every unit of work)

A "unit" = one file or one small module, never "the whole phase."

1. **`/clear`** to start clean (context bloat is the #1 failure). Load only what's relevant with `@docs/…`.
2. **Plan mode** (Shift+Tab twice) for anything non-trivial. Ask Claude Code to state the plan, file paths, and how it'll verify. Review it. Correct misunderstandings *now*, cheaply.
3. **Write the contract/test first** where it makes sense (the `Finding` shape, the metric's expected outputs, the adapter interface). Tests are how you get "verify by running" for free.
4. **Implement one slice.** Small diff. If it wants to touch five files, stop it and narrow.
5. **Run it.** `pytest tests/test_x.py -q` or the actual command. Paste output. No green = not done.
6. **Read the diff yourself.** Don't merge what you didn't look at.
7. **Commit** (see git strategy). One working increment = one commit.
8. **Tick `PROGRESS.md`** and note anything learned. This is the memory that survives `/clear`.

Golden phrasing to start a phase: *"Read @docs/ARCHITECTURE.md and @docs/PROGRESS.md. We're on Phase N, task T-N.x. Plan it in plan mode first — file paths, the test you'll write, and how you'll verify. Don't code yet."*

---

## Build principle: vertical slices, not horizontal layers

Do **not** build all the adapters, then all the modules, then all the reports. Build the **thinnest end-to-end path first**, prove it runs, then widen. The first slice (Phase 1) walks: corpus → embed → Chroma → sample → invert → score → print. Once that one path is green, everything else is an extension of a working spine, and every later addition is testable against something real.

---

## Complete file manifest (in build order)

Legend — **You** = your judgment/curation is the point; **CC** = Claude Code writes, you review; **CC+test** = write the test first.

### Slice 0 — Scaffolding (do once, fast)
| File | Purpose | Who |
|---|---|---|
| `requirements.txt` | pinned deps (`transformers==4.30.2`, `accelerate==0.21.0`, `torch`, `sentence-transformers`, `chromadb`, `pyyaml`, `pytest`) | CC |
| `.gitignore` | ignore `.venv/`, `__pycache__/`, `results/`, `*.sqlite3`, HF cache | CC |
| `pyproject.toml` | make `leaklens` an installable package; enable `python -m leaklens` | CC |
| `leaklens/__init__.py` | package marker | CC |
| `leaklens/_compat.py` | Windows `resource`-module stub + force-CPU helper; imported first everywhere | CC |
| `tests/conftest.py` | shared fixtures (tiny in-memory corpus, temp Chroma dir) | CC |
| `README.md` (stub) | one-paragraph placeholder; filled in Phase 6 | CC |

### Slice 1 — Inversion pipeline (Phase 1, the spine)
| File | Purpose | Who |
|---|---|---|
| `leaklens/inversion/inverter.py` | load vec2text once; `invert(embeddings) -> list[str]`; CPU-safe, batched | CC (from smoke test) |
| `tests/test_inverter.py` | golden: a known low-entropy sentence recovers its key words | CC+test |
| `corpus/corpus.jsonl` | ~200–300 labelled sentences: `{text, type, key_entities}` | **You** (curate) |
| `corpus/build_corpus.py` | generate candidates + validate schema/labels | CC, **you** curate output |
| `leaklens/adapters/base.py` | `VectorStoreAdapter.sample(n) -> [(vector, text|None)]` | CC |
| `leaklens/adapters/chroma_adapter.py` | Chroma implementation | CC+test |
| `tests/test_chroma_adapter.py` | round-trips vectors + labels | CC+test |
| `leaklens/inversion/metrics.py` | recovery = key-entity recall + similarity ≥ threshold | **You** define, CC implements |
| `tests/test_metrics.py` | metric matches your definition on hand-labelled pairs | CC+test |
| `scripts/embed_corpus.py` | embed `corpus.jsonl` → a Chroma store (one-time) | CC |
| `scripts/run_inversion_demo.py` | the spine: store → sample → invert → score → print side-by-side | CC |

*Slice-1 done = the demo prints original↔recovered with scores, and you can defend the metric in two sentences.*

### Slice 2 — The defense tradeoff study (Phase 2, the finding)
| File | Purpose | Who |
|---|---|---|
| `leaklens/inversion/defenses.py` | Gaussian noise (per σ), token/mean pooling, quantization | CC+test |
| `tests/test_defenses.py` | defenses transform vectors as specified | CC+test |
| `leaklens/inversion/retrieval.py` | `recall_at_k` on a store (utility measurement) | CC+test |
| `studies/defense_sweep.py` | for each defense level: leakage + recall@k; save `results/sweep.json` | **You** design, CC codes |
| `studies/plot_tradeoff.py` | render the privacy/utility curve → `results/tradeoff.png` | CC |

*Runs heavy → free Colab T4 (same code). Slice-2 done = the curve exists + a one-paragraph finding in your words.*

### Slice 3 — The tool wrapper (Phase 3)
| File | Purpose | Who |
|---|---|---|
| `leaklens/finding.py` | `Finding` dataclass + `Verdict`/`Severity` enums | CC+test |
| `tests/test_finding.py` | serialization + defaults | CC+test |
| `leaklens/config.py` | load/validate `config.yaml`; enforce ownership gate | CC+test |
| `config.example.yaml` | documented example config | CC |
| `leaklens/modules/base.py` | `Module.run(target, cfg) -> Finding` contract | CC |
| `leaklens/modules/inversion.py` | wraps Slice-1 pipeline → `Finding`; **honest-failure path** (no inverter → INCONCLUSIVE) | CC |
| `tests/test_module_inversion.py` | incl. no-inverter → INCONCLUSIVE, never a fake score | CC+test |
| `leaklens/report/owasp.py` | `Finding` → OWASP LLM Top-10 category | CC |
| `leaklens/report/scorecard.py` | live terminal scorecard | CC |
| `leaklens/report/json_report.py` | persisted JSON | CC |
| `leaklens/report/html_report.py` | persisted HTML (single file, no framework) | CC |
| `leaklens/runner.py` | orchestrate enabled modules; isolate each in try/except → INCONCLUSIVE | CC+test |
| `tests/test_runner.py` | a crashing module becomes INCONCLUSIVE; others still run | CC+test |
| `leaklens/cli.py` + `leaklens/__main__.py` | `python -m leaklens scan --config …` | CC |

*Slice-3 done = one command runs end-to-end on a Chroma store and writes terminal + HTML/JSON reports.*

### Slice 4 — Validation harness (Phase 3.5, the moat)
| File | Purpose | Who |
|---|---|---|
| `scripts/build_validation_targets.py` | build ≥3 leaky + ≥3 hardened stores (raw vs noised/pooled; loose vs tight) | **You** design |
| `leaklens/validation/harness.py` | run the tool over all targets; assert flags-leaky / clears-hardened | CC |
| `tests/test_validation.py` | the matrix classifies correctly and reproducibly (fixed seed) | CC+test |
| `results/validation_matrix.md` | generated 6-target table — the best README artifact | CC |

### Slice 5 — Surface 2: cache poisoning (Phase 4)
| File | Purpose | Who |
|---|---|---|
| `scripts/setup_gptcache.py` | stand up a self-hosted GPTCache | CC |
| `leaklens/adapters/gptcache_adapter.py` | plant/query cache entries | CC+test |
| `leaklens/adapters/faiss_adapter.py` | FAISS store (nearly free → honest multi-store) | CC+test |
| `leaklens/modules/cache_poison.py` | plant benign canary, graded paraphrases, poison radius → `Finding` | **You** design, CC codes |
| `tests/test_cache_poison.py` | loose threshold flagged, tight cleared | CC+test |

### Slice 6 — Stretch: Ghost Vectors (Phase 5, cut freely)
| File | Purpose | Who |
|---|---|---|
| `leaklens/modules/ghost_vectors.py` | plant → soft-delete → show still recoverable | CC |
| `tests/test_ghost_vectors.py` | recovery post-delete, or honest "store hard-deletes" | CC+test |

### Slice 7 — Polish (Phase 6)
| File | Purpose | Who |
|---|---|---|
| `README.md` (final) | leads with the infra-layer gap; quickstart; the validation table; a demo GIF | **You** + CC |
| `docs/mechanisms.md` | each attack explained **in your own words** (whiteboard defense) | **You** |
| `results/demo.gif` | recorded run | **You** |

---

## Testing strategy

- **Contract tests** first for `Finding`, the adapter interface, the module contract.
- **The metric is the most important test** — hand-label ~10 original↔recovered pairs, assert your metric scores them the way you'd defend out loud. If you can't write that test, you don't yet have a defensible metric.
- **The honest-failure test is mandatory**: no-inverter target → `INCONCLUSIVE`, never a number.
- **Golden corpus tests** use a tiny fixed subset so they run fast on CPU.
- Keep tests fast: mock the heavy inverter where you're testing *plumbing*, use the real one only in the few golden inversion tests.

## Git strategy

- One branch per slice: `slice/1-inversion-pipeline`, etc.
- Commit per working increment (conventional commits: `feat:`, `test:`, `fix:`, `docs:`). A commit should leave tests green.
- Never commit `.venv/`, the HF cache, `results/`, or any real target data.
- Tag the end of each slice (`v0.1-slice1`) so you always have a demoable point.

## Documentation discipline (keeps docs from drifting)

- `CLAUDE.md` stays small and stable — only rules that prevent mistakes. If it grows past ~200 lines, push detail into `docs/` and reference with `@`.
- `PROGRESS.md` is the living checklist — update it every session; it's the memory that survives `/clear`.
- If a locked decision changes, update `DECISIONS.md` in the same commit — the decision log must never lie.
- Docstrings say *why*. The *what* is in the code.

## Environment gotchas (already learned in Phase 0)

- Python **3.11** (not 3.14 — too new for these packages).
- `pip install "transformers==4.30.2" "accelerate==0.21.0"` **after** other installs so the pin wins.
- Windows: the `resource`-module stub must run before `import vec2text` → put it in `leaklens/_compat.py` and import that first.
- Inversion ≈ 48 s/sentence on CPU at `num_steps=20`; use `num_steps=5` for quick checks, full steps for real runs, Colab for sweeps.
