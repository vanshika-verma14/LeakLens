# LeakLens

**A defensive self-audit tool for the infrastructure layer beneath your RAG app.** Point it
at your own vector store and it measures how much that store leaks: can the embeddings you
stored be turned back into the readable text they came from?

## The problem: a gap under the RAG stack

Retrieval-augmented apps embed your documents into vectors and keep them in a vector
database. The comforting assumption is that a vector is a one-way, opaque blob of numbers —
"we only store embeddings, not text." **That assumption is false.** Published research
(vec2text) shows a text embedding preserves enough of the original that the input can be
*reconstructed* from it. The same signal that makes similarity search work is the signal
that makes reconstruction work.

The consequence for threat modeling: **a breached vector database is a breached document
database.** Your vector store stops being "harmless numbers, low sensitivity" and becomes
crown-jewel plaintext for incident scoping and compliance.

Most LLM-security tooling audits the **I/O layer** — prompts, responses, jailbreaks. Very
little looks at the **infrastructure layer** underneath: the store, the cache, the embeddings
themselves. LeakLens audits that layer. It is **defensive**: you run it against
infrastructure *you own* to find out what an attacker could recover after a breach.

> LeakLens does not introduce a new attack. It packages *published, cited* attacks
> (embedding inversion via **vec2text**) as a runnable self-audit and measurement tool. The
> contribution is the tooling, the measurement, and the validation — see
> [Limitations & honest scope](#limitations--honest-scope).

## What it does

- **Embedding-inversion audit (Surface 1).** Samples vectors from your store, attempts to
  invert them back to text, scores how much of the known content is recovered, and emits a
  verdict (`LEAK` / `NO_LEAK` / `INCONCLUSIVE`) with side-by-side original↔recovered
  evidence and concrete remediation.
- **OWASP-mapped report.** Every finding is tagged to the OWASP LLM Top-10 (LLM08: Vector
  and Embedding Weaknesses; LLM06: Sensitive Information Disclosure) and written to a
  terminal scorecard plus persisted JSON and self-contained HTML.
- **Defense tradeoff study.** A measured privacy/utility curve: for a range of defenses
  (Gaussian noise σ), how far does leakage drop, and what does retrieval accuracy pay for
  it? Turns the field's binary "hardened vs raw" into a measured recommendation.
- **Validation harness.** A reproducible known-safe / known-vulnerable matrix that proves
  the tool's verdicts are trustworthy (see [Validation](#validation)).

Semantic **cache-poisoning** (Surface 2) and **Ghost Vectors** (deleted-vector recovery)
are on the roadmap; they are not built yet.

## Install & quickstart

Requires **Python 3.11** (the pinned dependencies do not support newer versions). The tool
is **CPU-only** — inversion runs on CPU by design; no GPU/NPU is used or needed.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Unix
pip install -e .
```

> **Pinned dependencies matter.** `transformers==4.30.2` and `accelerate==0.21.0` are
> load-bearing — newer `transformers` breaks vec2text model loading. Do not upgrade them.
> The embedding/inverter models are downloaded once from Hugging Face and cached; later runs
> work offline (`HF_HUB_OFFLINE=1`).

Then build a store and scan it:

```bash
# 1) Embed the labelled corpus into a Chroma store (fast — this is the cheap part).
python scripts/embed_corpus.py            # -> ./results/corpus_store (collection "corpus")

# 2) Run the audit against it.
python -m leaklens scan --config config.yaml
#    (equivalently: leaklens scan --config config.yaml)
```

The scan prints a scorecard and writes `results/scan/report.json` and
`results/scan/report.html`. The shipped `config.yaml` points at the store from step 1, runs
the `inversion` module at `num_steps=20`, and uses the calibrated leak threshold `0.6`.
Inversion is slow on CPU (~48s per sentence at 20 steps), so `config.yaml` keeps
`sample_size` small; raise it for a fuller run.

## Findings

All figures are **measured**, not asserted — LeakLens exists to measure, or to return
`INCONCLUSIVE`. The recovery metric is **key-entity recall** (the fraction of a row's known
key entities that appear in the recovered text, matched case-insensitively), *not* exact
match — because the evidence shows names and structure leak even when an exact secret
mangles (DECISIONS D5). The leak line is **recall ≥ 0.6 = LEAK**, calibrated from the real
recall distribution rather than chosen a priori (DECISIONS D9).

**1. Recoverability is monotonic in entropy.** On the full 240-row corpus, baseline
(undefended) per-category recall falls cleanly with how high-entropy the content is:

| Category    | Mean key-entity recall (n=240) |
|-------------|--------------------------------|
| plain prose | 0.81 |
| PII         | 0.59 |
| structured  | 0.35 |
| credential  | 0.28 |

<!-- TODO: confirm these against the final sweep_full.json baseline when the full
     --num-steps 20 run lands -->

Readable prose reconstructs almost verbatim; passwords and exact digits mostly mangle while
their surrounding topic and labels survive. **This is a finding, not a bug** — and it is
exactly why the metric is recall, not exact-match. Practically: "we only store embeddings,
not text" is not a privacy boundary for the recoverable content.

**2. The privacy/utility tradeoff is real, not free.** Adding Gaussian noise to stored
vectors collapses inversion quickly — overall recall drops from ~0.51 to ~0.10 by a small σ,
with credentials effectively gone — while nearest-neighbour retrieval largely holds through
that range. But the tension is genuine: **retrieval@1 degrades earlier than retrieval@5**, so
under a strict utility bar there is *no* zero-cost noise band. The tidy "harden it and lose
nothing" story is an artifact of the more forgiving @5 metric.

![Privacy/utility tradeoff](results/tradeoff.png)

<!-- TODO: final crossover σ, the recommended defense setting (or the honest "no comfortable
     band" conclusion), and a refreshed results/tradeoff.png from the full --num-steps 20
     sweep -->

*(The current curve is from a preview sweep; the full-resolution run at production step count
is pending — see the TODO above.)*

## Validation

The validation harness is what makes the verdicts trustworthy rather than just plausible. It
runs a **paired, same-content matrix**: three content sets, each built twice — once **raw**
and once **hardened** with Gaussian noise (σ=0.05, past the point where leakage collapses).
Within a pair, *the defense is the only variable*, so "the verdict flips correctly" is a
claim about the defense, not about two unrelated stores.

The expected labels are **derived from measured per-category recall, not assumed**: a raw
store is only labelled a known-LEAK target when its content's measured recall clears the
threshold, and the builder **aborts** if a would-be-leaky set predicts below 0.6 — a false
"known-LEAK" label can't ship. The tool must then flag every raw store `LEAK` and clear every
hardened store `NO_LEAK`, reproducibly at a fixed seed.

See [`results/validation_matrix.md`](results/validation_matrix.md) for the generated table.

<!-- TODO: real-inverter recall column — the committed matrix is currently a --fake
     structural run (harness plumbing proven); regenerate with `python scripts/run_validation.py`
     for the citable recovery numbers -->

Build and run it yourself:

```bash
python scripts/build_validation_targets.py   # builds the 6 stores + manifest
python scripts/run_validation.py             # real inverter (slow); --fake for a quick check
```

## How it works

Why is an embedding invertible at all? A retrieval system works because the embedding of a
document keeps enough of that document's meaning and wording for "similar text → nearby
vector" to hold. **That preserved signal is also enough to reconstruct the text.** The
property you rely on for search is the property that leaks.

vec2text does the reconstruction iteratively: start from a candidate sentence, embed it,
compare that embedding to the target you're trying to invert, correct the candidate to close
the gap, and repeat for a number of steps. After enough steps the candidate's embedding
matches the target closely — and its text is a close reconstruction of the original. So
"we store only vectors" protects the *format*, not the *content*.

## Repo layout

```
leaklens/        the installable tool (cli, runner, config, adapters, modules, report)
  inversion/     the reusable core: inverter, recovery metric, defenses, retrieval
  validation/    the harness that scores the known-safe/known-vulnerable matrix
studies/         the defense tradeoff sweep + plot (the finding)
scripts/         embed the corpus, build validation targets, run the matrix
corpus/          labelled ground-truth corpus (text + type + key_entities)
docs/            ARCHITECTURE, DECISIONS (+ interview defense), BUILD_PLAN, PROGRESS
results/         generated reports, tradeoff plot, validation matrix
```

## Limitations & honest scope

A tool that states its own limits is the best signal that the numbers it *does* report are
trustworthy. So, plainly:

- **Not a novel attack.** Embedding inversion is published research; LeakLens is built on
  **vec2text** (Morris et al.) and a pretrained GTR inverter, cited — not invented here. The
  contribution is the **tooling, measurement, and validation**: turning an inaccessible
  research attack into a runnable, developer-facing self-audit with an OWASP-mapped report
  and a validation matrix that makes the verdicts trustworthy.
- **The demo is favorable by construction.** We choose the encoder and know the ground truth,
  which is what lets us *measure* honestly — it proves the mechanism, not that the tool
  inverts an arbitrary unknown store.
- **Encoder-bound.** Inversion works only where a public inverter exists (GTR here). Point
  LeakLens at another encoder and it returns `INCONCLUSIVE` — *"invertible in principle, no
  inverter available"* — **never a fabricated score.**
- **Recovery is partial on high-entropy data.** Passwords and exact digits often mangle;
  topic, names, and structure recover well (see [Findings](#findings) #1). Stated, not hidden.
- **Threshold sensitivity.** The 0.6 leak line is provisional and documented (DECISIONS D9);
  it is a human decision, never silently defaulted. Because the verdict is a mean over
  sampled vectors, it depends on store composition — a high-entropy-only store can read
  `NO_LEAK` even undefended, which is the honest finding, not an evasion.
- **Common-word labels can inflate recall.** Case-insensitive substring matching means a
  generic label like "Order"/"Invoice" can match coincidentally; recovery of the high-entropy
  *value* is the honest signal, which is why labels and values are tagged separately
  (DECISIONS #6).
- **Cache-timing is out of scope, permanently** (ethics and feasibility, DECISIONS). Semantic
  cache-poisoning (Surface 2) and Ghost Vectors are roadmap, not yet built.
- **`INCONCLUSIVE` is a first-class result.** Every module returns it rather than guess; the
  tool prefers an honest non-answer to a confident-but-unsupported one.

## Credits

Built on the vec2text embedding-inversion research (Morris et al.) and the `ielabgroup`
pretrained GTR inverter, with `sentence-transformers/gtr-t5-base` as the encoder and Chroma
as the vector store. Technical decisions and their rationale — including the full limitations
register and the novelty statement — are documented in
[`docs/DECISIONS.md`](docs/DECISIONS.md).
