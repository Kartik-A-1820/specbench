# Results

> **Note:** This file will be populated after running the benchmark.
> Run `python scripts/run_phase1_bench.py` (Phase 1) and
> `python scripts/run_phase2_sweep.py` (Phase 2) to collect measurements.
> Every number here must trace back to a row in `results/raw_measurements.jsonl`.

---

## Phase 1: Vanilla Speculative Decoding

### Configuration

| Parameter | Value |
|---|---|
| Target model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Draft model | JackFram/llama-68m |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Max new tokens | 128 |
| Prompt set | 30 prompts (prose + code + instructional) |
| Warmup discarded | 2 runs |
| Context length | 2048 tokens |

### Speedup by K

| K | Baseline (tok/s) | Spec (tok/s) | Speedup | 95% CI | Mean α |
|---|---|---|---|---|---|
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Latency Percentiles (K=4)

| Metric | Baseline | Speculative |
|---|---|---|
| p50 inter-token (ms) | _TBD_ | _TBD_ |
| p90 inter-token (ms) | _TBD_ | _TBD_ |
| p99 inter-token (ms) | _TBD_ | _TBD_ |

Speculative decoding exhibits bursty latency: several tokens are emitted
at once following a successful verify pass, then the system pauses for the
next round. The distribution is bimodal; p99 captures the tail of this
pattern. See `docs/methodology.md` for discussion.

### Acceptance Rate by Domain

| Domain | Mean α (K=4) | Notes |
|---|---|---|
| Prose | _TBD_ | — |
| Code | _TBD_ | Higher acceptance expected due to syntactic constraints |
| Instructional | _TBD_ | — |

---

## Phase 2: EAGLE Draft Head

### Training Details

| Parameter | Value |
|---|---|
| Training corpus | Built-in corpus (~500K tokens) |
| Hidden state cache | `results/hidden_states/` |
| Training epochs | 3 |
| Optimiser | AdamW, lr=3e-4 |
| Loss | 0.1 × SmoothL1 + 0.9 × CrossEntropy |
| Peak VRAM during training | _TBD_ (target unloaded) |

Loss curves are in `results/figures/eagle_loss_curves.png`.

### Sweep Results

Full results are in `results/pareto.csv`.

| Configuration | K | Mean α | Speedup | 95% CI |
|---|---|---|---|---|
| L1 I5632 (EAGLE) | 4 | _TBD_ | _TBD_ | _TBD_ |
| L1 I2816 (EAGLE) | 4 | _TBD_ | _TBD_ | _TBD_ |
| L2 I5632 (EAGLE) | 4 | _TBD_ | _TBD_ | _TBD_ |
| L2 I2816 (EAGLE) | 4 | _TBD_ | _TBD_ | _TBD_ |
| llama-68m (Phase 1) | 4 | _TBD_ | _TBD_ | _TBD_ |

### Pareto Curve

See `results/figures/pareto_acceptance_vs_speedup.png`.

The curve reveals the speedup-vs-acceptance tradeoff: deeper speculation
raises acceptance length but costs more draft forward passes per round, so
wall-clock speedup peaks at some K and then declines. The optimal K is
hardware-dependent; it is measured empirically here rather than assumed.

---

## Data Provenance

Every number in this document corresponds to rows in `results/raw_measurements.jsonl`.
To reproduce any statistic:

```python
import json, numpy as np
from specbench.stats import bootstrap_speedup_ci

records = [json.loads(l) for l in open("results/raw_measurements.jsonl")]
baseline = [r for r in records if r["mode"] == "baseline"]
spec = [r for r in records if r["mode"] == "spec" and r["K"] == 4]
baseline_tps = np.array([r["tokens_per_s"] for r in baseline])
spec_tps = np.array([r["tokens_per_s"] for r in spec])
speedup, ci_lo, ci_hi = bootstrap_speedup_ci(baseline_tps, spec_tps)
print(f"{speedup:.2f}× (95% CI {ci_lo:.2f}–{ci_hi:.2f})")
```
