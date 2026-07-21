# LeakLens — Validation Matrix 

> _(--fake structural run: harness plumbing only, not real recovery numbers.)_

Known-safe / known-vulnerable matrix. Each **leaky** store holds raw embeddings; its **hardened** pair holds the *same content* with Gaussian noise (sigma=0.05) baked in — so the defense is the only variable within a pair. A store is flagged LEAK when mean key-entity recall >= 0.6 (the calibrated threshold, DECISIONS D9).

Expected labels are **derived from measured per-category recall**, not assumed: a raw store is labelled LEAK only when its content's measured recall exceeds the threshold. The tool must reproduce these labels.

**Result: 6/6 correctly classified** (reproducible at seed=42).

| Target | Content | Defense | Expected | Verdict | Mean recall | Pass |
|---|---|---|---|---|---|---|
| setA-raw | 20 plain | none | LEAK | LEAK | 1.000 | PASS |
| setA-noised | 20 plain | gaussian sigma=0.05 | NO_LEAK | NO_LEAK | 0.000 | PASS |
| setB-raw | 20 plain | none | LEAK | LEAK | 1.000 | PASS |
| setB-noised | 20 plain | gaussian sigma=0.05 | NO_LEAK | NO_LEAK | 0.000 | PASS |
| setC-raw | 12 plain + 8 pii | none | LEAK | LEAK | 1.000 | PASS |
| setC-noised | 12 plain + 8 pii | gaussian sigma=0.05 | NO_LEAK | NO_LEAK | 0.000 | PASS |
