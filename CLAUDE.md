# CLAUDE.md — LeakLens

LeakLens is a **defensive CLI** that audits the infrastructure under RAG apps for leakage:
(1) embedding-inversion of a vector store, (2) semantic-cache poisoning. It packages *published*
attacks as a *self-audit* tool + measurement study. It is engineering + measurement, not a new attack.

Full context (read on demand, don't inline): @docs/ARCHITECTURE.md · @docs/DECISIONS.md · @docs/BUILD_PLAN.md · @docs/DEVELOPMENT.md
Live task state / checklist: @docs/PROGRESS.md

## TIER 1 — Hard rules (violating any of these is a workflow failure)
- **Plan before non-trivial work.** If a task is >3 steps or touches architecture, enter plan mode and show the plan BEFORE editing. If multiple interpretations exist, surface them — don't silently pick.
- **One vertical slice at a time.** No large multi-file generation. Build the smallest end-to-end path, run it, then expand.
- **Verify by running.** Never say "done" without evidence: a passing test or actual pasted output. Read the diff.
- **Every module returns a `Finding`** (see ARCHITECTURE). It catches its own errors and returns `INCONCLUSIVE` — it must never crash the runner.
- **`INCONCLUSIVE` is a valid, first-class result.** Prefer it to a confident-but-unsupported verdict. Never fabricate a recovered string or a score.

## Tech stack — do NOT change without updating DECISIONS.md
- Python 3.11 in `.venv`. **CPU-only** — the AMD iGPU and NPU are unusable for torch; do not attempt GPU/NPU.
- **Pinned: `transformers==4.30.2`, `accelerate==0.21.0`.** Do NOT upgrade — newer transformers breaks vec2text loading.
- On Windows, import `leaklens._compat` FIRST in any entrypoint that uses vec2text (it stubs the Unix-only `resource` module).
- Encoder `sentence-transformers/gtr-t5-base` · Inverter `ielabgroup/vec2text_gtr-base-st_{inversion,corrector}` · Store Chroma (primary) + FAISS · Cache GPTCache.

## Commands
- Activate venv: `.venv\Scripts\activate`
- Tests: `pytest tests/ -q`  (prefer single tests while iterating: `pytest tests/test_x.py -q`)
- Run tool: `python -m leaklens scan --config config.yaml`

## Conventions
- Recovery metric = **key-entity recall + semantic-similarity threshold**, reported together. NOT exact-match. (Why: DECISIONS D5.)
- Ownership gate: any module that writes state or probes shared infra requires `i_own_this_target: true`.
- Reproducibility: thread `seed` everywhere; a fixed seed must reproduce the same validation verdicts.
- Small pure functions, type hints, docstrings that explain *why* not *what*. Keep it simple — no frameworks we don't need.

## TIER 1 — Never do these
- Never present any attack as novel or "ours" — all are cited research (DECISIONS).
- Never build the **cache-timing** module — it is deliberately dropped scope. Do not re-add it.
- Never upgrade `transformers`; never route to GPU/NPU.
- Never let the recovery **threshold** be chosen/hardcoded silently — it is a human decision to be documented.
- Never fabricate recovered text or scores — measure, or return `INCONCLUSIVE`.
- Never target infrastructure the user does not own.
