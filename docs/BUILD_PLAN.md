# LeakLens — Build Plan

*The execution roadmap: what we build, in what order, who does each part, and when a phase is "done." Read `ARCHITECTURE.md` for the system design and `DECISIONS.md` for locked technical calls + interview defense.*

---

## What LeakLens is (one paragraph)

A defensive, command-line **self-audit tool** for the infrastructure layer beneath RAG/LLM apps. A developer points it at their own vector store (and optionally their semantic cache); it measures how much that infrastructure leaks — **can stored embeddings be turned back into readable text?** and **can the semantic cache be poisoned?** — and returns an OWASP-mapped leakage report with concrete remediation. The attacks are published research (cited, not claimed as ours); the contribution is **operationalization + measurement + a validation harness that makes the verdicts trustworthy.**

**Status:** Smoke test PASSED (2024 GTR encoder + ielabgroup vec2text inverter round-trips on the target CPU laptop, ~48s/sentence, word-perfect on low-entropy text). Core feasibility is proven. We are cleared to build.

---

## Scope (locked)

| Tier | Item | Status |
|---|---|---|
| **Core** | Surface 1: embedding-inversion leakage + **defense tradeoff study** | Build deep |
| **Core** | Tool wrapper: CLI, config, Chroma + FAISS adapters, HTML/JSON report, OWASP map | Build solid |
| **Core** | Validation harness (known-safe vs known-vulnerable matrix) | Build — this is the moat |
| **Core** | Honest self-failure reporting (no inverter → say so, don't fake) | Build — cheap, high credibility |
| **Breadth** | Surface 2: semantic-cache poisoning auditor (GPTCache) | Build if Surface 1 is fully done |
| **Stretch** | Ghost Vectors: deleted-vector still-recoverable check | Bonus; cut without guilt |
| **DROPPED** | Cache-timing module | Out — can only prove a self-configured toy; ethics |
| **DROPPED** | Scanning public RAG demos | Out — they don't expose their stores to us |
| **DROPPED** | Any zero-shot / "works on any encoder" / "Nmap for all AI infra" claim | Out — overclaim |

**Depth-over-breadth rule:** if Surface 2 starts going shallow, ship Surface 1 deep and list the rest as roadmap. A finished single-surface tool with a real finding beats a half-finished multi-surface one every time — the interview tests depth.

---

## Execution order (and why this order)

We build toward a **defensible finding as fast as possible**, then wrap it in a tool. The finding (Phase 1–2) is the un-fakeable, resume-worthy core and the thing that survives if you stop early. The CLI is a wrapper around a *working, validated* pipeline — don't build the shell before the thing it wraps works.

```
Phase 0  Smoke test + env ............... DONE
Phase 1  Core pipeline + corpus + metric  ── the ground truth
Phase 2  Defense tradeoff STUDY ......... ── the finding (highest value)
Phase 3  Tool wrapper (CLI/report/adapters)
Phase 4  Surface 2 (cache poisoning)
Phase 5  Stretch: Ghost Vectors
Phase 6  Polish: README, writeups, matrix, demo GIF
```

If you must stop after any phase, the project is still coherent: after Phase 2 you have "an empirical study of embedding-inversion defenses"; after Phase 3 you have "a working audit tool"; after Phase 4, "a two-surface auditor."

---

## Phases in detail

### Phase 0 — Smoke test + environment — ✅ DONE
- Python 3.11 venv, `torch` + `vec2text` + `sentence-transformers`, pinned `transformers==4.30.2`, Windows `resource`-module stub.
- Confirmed round-trip on CPU. **Outcome recorded:** low-entropy text recovers verbatim; high-entropy secrets (passwords, exact digits) mangle while names/structure survive. → This directly sets the recovery-metric design in Phase 1.

### Phase 1 — Core inversion pipeline + ground-truth corpus + recovery metric
**Goal:** a real pipeline (not a script) that embeds a labelled corpus into Chroma, samples vectors, inverts them, and scores recovery against known ground truth.
- **T1.1** Build the labelled corpus: ~200–300 sentences in `corpus.jsonl`, each tagged with type (`plain`, `pii`, `secret`, `credential`) and the key entities it contains. Mix of normal text + planted fake secrets. *(You own the design; Claude Code can generate candidate sentences you curate.)*
- **T1.2** Embed corpus with `gtr-t5-base` → load into a Chroma collection. *(Claude Code writes; you verify counts.)*
- **T1.3** Inversion wrapper: clean module around `vec2text.invert_embeddings` (load once, batch, CPU-safe). *(Claude Code, from the working smoke test.)*
- **T1.4** **Recovery metric** (YOURS to define + defend): key-entity/keyword recall **+** semantic similarity above a justified threshold, reported together. NOT exact-match (Phase-0 evidence shows why: structure/names leak even when the secret mangles). Document the threshold choice.
- **DONE when:** `pipeline.py` takes the Chroma store → returns a per-sentence recovery score + side-by-side original↔recovered, and you can defend the metric in two sentences.

### Phase 2 — The defense tradeoff study (**the finding — highest value**)
**Goal:** the killer artifact — a measured privacy/utility curve, not a binary.
- **T2.1** Implement defenses in `defenses.py`: Gaussian noise at several σ, token/mean pooling, optionally dimensionality reduction or quantization.
- **T2.2** For each defense level, measure **two** things on the same store: **leakage** (recovery metric) and **utility** (retrieval recall@k — does the RAG still find the right doc?).
- **T2.3** Run the sweep (heavy → free Colab T4; same code, minutes instead of hours) and produce the **tradeoff curve** + a recommended defense setting (or an honest "no comfortable middle exists for this encoder").
- **DONE when:** a plot/table shows the crossover where inversion collapses but retrieval survives, saved to `results/`, with a one-paragraph finding written in your own words.

### Phase 3 — Tool wrapper (makes it software, not a notebook)
**Goal:** the thing a developer actually runs. *(Claude Code does most of this fast.)*
- **T3.1** `Finding` dataclass + module contract (see ARCHITECTURE).
- **T3.2** `config.yaml` loader + the **ownership gate** (`i_own_this_target: true`).
- **T3.3** Adapters: **Chroma** (primary) + **FAISS** (nearly free to add → honest "multi-store").
- **T3.4** `inversion` module wraps Phase 1–2 behind the contract; emits a `Finding` with score/severity/evidence/remediation.
- **T3.5** Report: live terminal scorecard + persisted **HTML + JSON**, each finding mapped to **OWASP LLM Top-10** (LLM08 Vector & Embedding Weaknesses; LLM06 Sensitive Info Disclosure).
- **T3.6** **Honest self-failure:** point at an encoder with no inverter → returns `INCONCLUSIVE` "invertible in principle, no inverter available," never a fake score.
- **DONE when:** `leaklens scan --config config.yaml` runs end-to-end against a Chroma store and produces terminal + HTML/JSON reports.

### Phase 4 — Surface 2: semantic-cache poisoning auditor
**Goal:** confidentiality **and** integrity story. *(Build only once Phase 1–3 are solid.)*
- **T4.1** Stand up self-hosted **GPTCache** (pure-Python, CPU-fine).
- **T4.2** `cache_poison` module: plant a benign **labelled canary**, fire graded paraphrases at increasing semantic distance, measure **poison radius** + poisoned-hit rate vs. the cache threshold.
- **T4.3** Emit a `Finding` (loose threshold flagged, tight threshold cleared).
- **DONE when:** `--module cache-poison` reports a poison radius and correctly flags a loose-threshold cache while clearing a tight one.

### Phase 5 — Stretch: Ghost Vectors (deleted-vector recovery)
**Goal:** one distinctive, memorable finding. *(Cut freely if fiddly.)*
- Plant a secret → soft-delete it in the store → show it's **still reconstructible**. Based on the 2026 "Ghost Vectors" result.
- **DONE when:** the tool demonstrates recovery of a "deleted" entry, or you conclude the store hard-deletes and report that honestly.

### Phase 6 — Polish (what makes it read as real work)
- **T6.1** The **validation matrix**: ≥3 leaky + ≥3 hardened targets per surface; the tool must flag the leaky and clear the hardened. "6 targets, 3/3 correctly classified" is the single best README artifact.
- **T6.2** README leading with the *infrastructure-layer gap*, not "RAG security."
- **T6.3** A short **mechanism writeup per attack** in your own words (why retrieval signal = reconstruction signal; why cache locality fights collision-resistance). This is your whiteboard defense.
- **T6.4** A **demo GIF**.
- **DONE when:** a stranger can read the README, understand the gap, run one command, and see a trustworthy report.

---

## Realistic effort

| Phase | Focused effort | Claude Code leverage |
|---|---|---|
| 0 Smoke test | ✅ done | — |
| 1 Pipeline + corpus + metric | 2–4 days | some (glue) |
| 2 Defense study (**the finding**) | 3–5 days | little — the heart; Colab for sweeps |
| 3 Tool wrapper | 2–4 days | **a lot** |
| 4 Surface 2 | 3–5 days | some |
| 5 Ghost Vectors (stretch) | 2–3 days | some |
| 6 Polish | 2–3 days | some |

**Total:** ~2–3 weeks focused, or ~4–6 weeks part-time.
**Resume-strong minimum:** Phases 1–3 + validation (Surface 1 deep, working tool, honest-failure) ≈ 1.5–2 weeks — a complete project on its own.

---

## What is YOURS vs. what Claude Code does

**Yours (the parts an interviewer probes — do not outsource the judgment):**
- Defining and defending the recovery metric and its threshold.
- Interpreting results; spotting when a result is a bug vs. a finding.
- Corpus design and the validation matrix (what counts as a fair test).
- Explaining each mechanism in your own words at a whiteboard.
- The honesty calls (when to say INCONCLUSIVE).

**Claude Code (lean on it hard — this is correct tool use, not cheating):**
- CLI, config parsing, adapters, HTML/JSON report, plotting, GPTCache wiring, boilerplate, tests.

---

## Honest ceiling (never overclaim)

Not a novel research contribution — never imply it is. What it proves: you can read current (2023–2026) security papers, reproduce them, integrate them into working software, design an honest experiment, find a real tradeoff, and ship a tool that admits its limits. For a placement resume in security / ML-security, that sits well above the median ("I built a RAG chatbot").
