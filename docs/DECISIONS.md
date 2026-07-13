# LeakLens — Decisions & Defense

*Locked technical calls with the reason for each, the honest limitations register, and the novelty statement. This doubles as interview prep: every row is something you can be asked "why?" about.*

---

## Locked decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Language | **Python** | vec2text, embedding models, stats, and all the research repos are Python-first. Staying in Python keeps us closest to the code we reuse. |
| D2 | Embedding model | **`sentence-transformers/gtr-t5-base`** | Open, local, free (no API, no cost). A real RAG encoder (not a toy). Crucially, a released inverter exists for it. |
| D3 | Inverter | **`ielabgroup/vec2text_gtr-base-st_{inversion,corrector}`** | Pretrained — skips vec2text's ~5M-pair per-encoder training. **Confirmed working on the target CPU laptop** (~48s/sentence, 20 steps). |
| D4 | Vector store | **Chroma** primary, **FAISS** secondary | Chroma is trivial to stand up, embed, and sample. FAISS is nearly free to add and makes "multi-store" honest. pgvector/Pinecone = future. |
| D5 | Recovery metric | **Key-entity recall + semantic-similarity threshold** (NOT exact-match) | Phase-0 evidence: names/structure leak even when the exact secret mangles. Exact-match would wrongly score those as "safe." Report both signals. |
| D6 | Compute | **Develop on CPU laptop; heavy sweeps on free Colab T4** | Demo runs fine on CPU (~48s/sentence). The Phase-2 defense sweep over a full corpus is the only heavy part → Colab, same code, still ₹0. |
| D7 | Semantic-cache testbed | **GPTCache** | Pure-Python, CPU-friendly, the de-facto research testbed used across the poisoning papers. |
| D8 | Framing | **Defensive self-audit**, not offensive | The attacks are cited research inputs; our layer is measure + score + remediate. Differentiates from the offensive attack repos. |

### GPU/NPU note (target hardware: AMD Ryzen AI 7 350, Radeon 860M iGPU, NPU)
The iGPU (2 GB, shared RAM) and the NPU are **not usable for PyTorch** without major porting. Treat the machine as **CPU-only** — which is what every decision above already assumes. Do not spend time trying to accelerate on either; it's a multi-day detour for no project value.

---

## Dropped scope (and why time/effort doesn't rescue it)

| Dropped | Why it stays dropped |
|---|---|
| **Cache-timing side-channel module** | To get a positive LEAK you need a cache that shares across users; the only one you're *allowed* to touch (per the ownership gate) is one you configured to leak. Best case = "I proved my own toy leaks." Probing anyone else's cache breaks the ethics rules. Weeks of noisy timing stats for a weak result. |
| **Scanning public RAG demos** for a headline finding | Public demos don't expose their vector store to you — you can't sample vectors you can't reach. An access problem, not a time problem. |
| **Zero-shot / "works on any encoder" / "Nmap for all AI infra"** claims | We support the encoders we have inverters for. Claiming generality we can't back is the fastest way to lose credibility. Keep claims to exactly what's measured. |

---

## Limitations register (state these *before* anyone asks)

1. **Inversion demo is favorable by construction.** We choose the encoder and know the ground truth. That proves the *mechanism* and lets us *measure* honestly; it is not evidence the tool inverts arbitrary unknown stores. Stated plainly in the README.
2. **Recovery is partial on high-entropy data.** Passwords and exact digits often mangle; topic, names, and structure recover well. This is a *finding*, not a bug — and it's why the metric isn't exact-match.
3. **Encoder-bound.** Works where a public inverter exists (GTR here). Other encoders → the tool honestly reports "invertible in principle, no inverter available."
4. **Timing not covered.** Explicitly out of scope; listed as future work, not hidden.
5. **Not a novel attack.** All three attack families are published and cited. The contribution is operationalization + measurement + validation.

A tool that *states its own limits* is the strongest signal that the numbers it does report are trustworthy. Lead with this, don't bury it.

---

## Novelty statement (the defensible version)

**Not novel:** the attacks (embedding inversion, semantic-cache poisoning) and their underlying techniques (vec2text, similarity-collision). All published; all cited.

**Novel / defensible:**
- **Operationalization + unification** — no single developer-facing tool audits these infrastructure surfaces and emits a unified, OWASP-mapped leakage report. Turning research artifacts into one runnable self-audit scanner is the contribution.
- **Audit framing** — existing code is offensive (attack frameworks) or provider-side (Stanford's audit of vendors). LeakLens is defender-side self-audit: point it at *your own* app, get a verdict.
- **The measurement** — the privacy/utility tradeoff curve turns the field's binary "hardened vs raw" into a measured recommendation. You have to run the experiment to know its shape.
- **The validation harness** — a reproducible known-safe/known-vulnerable matrix that proves the verdicts are trustworthy.

**The one sentence to say in an interview:**
> "I took proven-but-inaccessible infrastructure attacks from recent papers and built the first developer-facing tool that audits your own AI app for them and scores the leakage — including the experiment design and validation matrix that prove the scores are trustworthy."

---

## Interview defense — the three questions you WILL get

1. **"Walk me through how embedding inversion actually works."**
   → A text embedding preserves enough semantic + lexical structure to support similarity search; that *same* preserved signal is enough to reconstruct the input. vec2text does it iteratively: generate a candidate, re-embed, compare to the target, correct, repeat for N steps. So "we only store embeddings, not text" is not a privacy boundary. *(Be able to draw this loop.)*

2. **"Couldn't an AI just build this?"**
   → AI wrote the plumbing (CLI, report, adapters). It could not choose or defend the recovery metric, design the validation matrix, decide when the result is a bug vs. a finding, or produce the tradeoff curve — those require running experiments and judgment. Point to the validation matrix and the honest INCONCLUSIVE cases: an AI-spammed project always claims success; measurement discipline shows in what it admits it can't do.

3. **"So this only matters after a breach?"**
   → Yes — and that's the point. Inversion reclassifies your vector store from "harmless numbers, low sensitivity" to "crown-jewel plaintext" for breach modeling and compliance. It changes how you defend the store, encrypt it, and scope an incident.

---

## Safety / ethics (non-negotiable)

- **Ownership gate** (`i_own_this_target: true`) required before any module that writes state or probes shared infra.
- **Benign canaries only** — planted poison entries are harmless, clearly-labelled, used only to *measure* radius.
- **Defensive framing throughout** — every offensive technique cited to its source paper; nothing presented as a novel attack.
- **No third-party production targets** in examples or defaults.
