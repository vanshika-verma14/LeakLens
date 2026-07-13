# LeakLens — Architecture

*The system design: repo layout, the core abstraction every module shares, the data flow, and the config schema. Deliberately thin and boring — the intelligence is in the measurement, not the plumbing.*

---

## Design principles

1. **One shape for every result.** Every module returns the same `Finding` object, so the runner and report never special-case a module.
2. **Modules are isolated.** One module failing or returning `INCONCLUSIVE` must never abort the others.
3. **Adapters hide the store.** The inversion module asks an adapter for "N sampled vectors (+ ground-truth text if available)"; it neither knows nor cares whether that's Chroma or FAISS.
4. **INCONCLUSIVE is first-class.** Every module can return it. Honesty over a confident-but-unsupported verdict.
5. **The study is not the tool.** The Phase-2 experiment lives in `studies/` as scripts; the tool in `leaklens/` reuses the same inversion + metric + defenses modules. Same code, two entry points.

---

## Repo layout

```
leaklens/
├── README.md                    # leads with the infra-layer gap
├── requirements.txt             # pinned: transformers==4.30.2, etc.
├── config.example.yaml
│
├── leaklens/                    # the installable tool
│   ├── __init__.py
│   ├── cli.py                   # `leaklens scan --config config.yaml`
│   ├── runner.py                # loads config, runs enabled modules, collects Findings
│   ├── finding.py               # the Finding dataclass (the shared shape)
│   ├── config.py                # parse + validate config.yaml; enforce ownership gate
│   │
│   ├── adapters/                # hide the store behind one interface
│   │   ├── base.py              # VectorStoreAdapter: sample(n) -> [(vector, text|None)]
│   │   ├── chroma_adapter.py    # primary
│   │   └── faiss_adapter.py     # secondary (nearly free → honest "multi-store")
│   │
│   ├── modules/                 # each returns a Finding
│   │   ├── base.py              # Module.run(target, cfg) -> Finding
│   │   ├── inversion.py         # Surface 1: wraps inversion/ behind the contract
│   │   └── cache_poison.py      # Surface 2: GPTCache canary auditor
│   │
│   ├── inversion/               # the reusable core (tool + studies share this)
│   │   ├── inverter.py          # loads vec2text once; invert_embeddings(); CPU-safe
│   │   ├── metrics.py           # recovery metric: key-entity recall + similarity
│   │   └── defenses.py          # gaussian noise, pooling, quantization
│   │
│   ├── report/
│   │   ├── scorecard.py         # live terminal output
│   │   ├── html_report.py       # persisted HTML
│   │   ├── json_report.py       # persisted JSON
│   │   └── owasp.py             # Finding -> OWASP LLM Top-10 category
│   │
│   └── validation/
│       └── harness.py           # runs the known-safe/known-vulnerable matrix
│
├── corpus/
│   └── corpus.jsonl             # labelled ground truth (sentence + type + key_entities)
│
├── studies/                     # the empirical work (Phase 2) — the finding
│   └── defense_sweep.py         # sweeps defenses; measures leakage AND recall@k; plots curve
│
├── results/                     # generated: reports, tradeoff plot, validation matrix
└── tests/
```

---

## The core abstraction: `Finding`

Every module produces exactly this. The report layer only ever consumes this.

```python
from dataclasses import dataclass, field
from enum import Enum

class Verdict(str, Enum):
    LEAK = "LEAK"
    NO_LEAK = "NO_LEAK"
    INCONCLUSIVE = "INCONCLUSIVE"

class Severity(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"; CRITICAL = "CRITICAL"

@dataclass
class Finding:
    module: str                       # "inversion" | "cache_poison"
    verdict: Verdict
    score: float                      # 0.0–1.0 leakage severity
    severity: Severity
    owasp: str                        # e.g. "LLM08: Vector and Embedding Weaknesses"
    summary: str                      # one line for the scorecard
    evidence: list = field(default_factory=list)   # original↔recovered pairs, numbers
    remediation: str = ""             # concrete fix (e.g. recommended noise σ)
    details: dict = field(default_factory=dict)    # raw metrics for the JSON report
```

**Module contract** (`modules/base.py`):

```python
class Module:
    name: str
    def run(self, target, cfg) -> Finding:
        """Isolated. Must return a Finding (including INCONCLUSIVE).
        Must not raise into the runner — catch, wrap as INCONCLUSIVE."""
```

The `runner` wraps each `run()` in a try/except so a crash becomes an `INCONCLUSIVE` Finding, never an aborted scan (satisfies FR-2 / FR-5).

---

## Data flow

```
config.yaml
    │
    ▼
[config.py]  parse + validate + ownership gate (i_own_this_target)
    │
    ▼
[runner.py]  for each enabled module:
    │           ├─ build target via [adapters/*]  (chroma | faiss | gptcache)
    │           ├─ module.run(target, cfg)  ── isolated, try/except → Finding
    │           └─ collect Finding
    ▼
[report/*]  scorecard (terminal)  +  html_report  +  json_report
                    └─ each Finding tagged via owasp.py
```

The **inversion module** internally calls: `adapter.sample(n)` → `inverter.invert()` → `metrics.score()` → assemble `Finding`. The **defense study** (`studies/defense_sweep.py`) calls the *same* `inverter`, `metrics`, and `defenses` — it just loops over defense levels and also measures `recall@k`. This shared core is why the tool and the study never diverge.

---

## Adapter interface

`sample(n)` returns a typed `Sample`, not a bare `(vector, text)` pair: recovery is scored
**per category**, so every sampled vector must carry back its `type` and `key_entities`.

```python
# adapters/base.py
@dataclass
class Sample:
    id: str
    vector: list[float]
    text: str | None = None          # None for a real breach target; present for our corpus
    type: str | None = None          # plain | pii | credential | structured
    key_entities: list[str] = field(default_factory=list)

class VectorStoreAdapter(ABC):
    def add(self, samples: list[Sample]) -> None: ...          # takes precomputed vectors
    def sample(self, n: int, *, seed: int | None = None) -> list[Sample]: ...  # seed = reproducible
    def count(self) -> int: ...
```

Adapters never embed text — they store precomputed vectors (the GTR encoder lives in
`inversion/inverter.py`). Chroma metadata can't hold a list, so `chroma_adapter` stores
`key_entities` as a JSON string and parses it back on read. Chroma and FAISS each
implement this; adding pgvector/Pinecone later = one new file, nothing else changes.

---

## `config.yaml` schema

```yaml
target:
  vector_store:
    type: chroma                 # chroma | faiss
    path: ./my_store
    collection: docs
    encoder: sentence-transformers/gtr-t5-base
  semantic_cache:                # optional (Surface 2)
    type: gptcache
    similarity_threshold: 0.8

modules:                         # which to run
  - inversion
  # - cache_poison

inversion:
  sample_size: 100
  num_steps: 20                  # vec2text correction steps (fewer = faster/rougher)
  recovery_threshold: 0.7        # the metric threshold you defined + defend

options:
  i_own_this_target: true        # SR-1 ownership gate — required to run write/probe modules
  output_dir: ./results
  seed: 42                       # reproducibility (NFR-3)
```

---

## Where each requirement lives (traceability)

| Req | Where |
|---|---|
| FR-1 config | `config.py` |
| FR-2 isolated modules | `runner.py` try/except |
| FR-3 terminal + HTML/JSON | `report/*` |
| FR-4 OWASP map | `report/owasp.py` |
| FR-5 INCONCLUSIVE | `Finding.verdict` + runner |
| FR-6 connect + sample | `adapters/*` |
| FR-7 recovery score + examples | `inversion/metrics.py`, `modules/inversion.py` |
| FR-8 graceful no-inverter | `modules/inversion.py` (honest-failure) |
| FR-9 poison radius | `modules/cache_poison.py` |
| SR-1 ownership gate | `config.py` |
| NFR-3 reproducibility | `options.seed` threaded everywhere |
