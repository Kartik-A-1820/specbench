# specbench

**An implementation of speculative decoding, with a correctness test suite, built to be benchmarked on a 4 GB laptop GPU.**

---

## Result

**1.321x** (95% CI 1.235–1.406) at K = 3 with ancestral sampling, and **1.442x**
(95% CI 1.373–1.510) at K = 2 greedy, against a target-only baseline of
11.40 tokens/s on a GTX 1650 Ti Mobile. Every figure is an interleaved paired
measurement over 29 scored prompts with bootstrap intervals; no point estimate
is reported without one.

The speedup is capped by acceptance, not by draft cost. The drafter contributes
about 18% of per-round cost at the optimum, so a drafter of literally zero cost
would still only reach 1.80x with this pair — the highest acceptance length
measured anywhere in the sweep. `llama-68m` and TinyLlama share the Llama
vocabulary and little else, and that shows up directly in α. The full
decomposition, including where this machine sits against its memory-bandwidth
roofline, is in the handover document.

A trained EAGLE draft head (Phase 2) reached **0.938x** — a genuine slowdown.
That is reported as measured, with the diagnosis, in `docs/results.md`.

To reproduce, double-click `run_all.bat` (or run it from a shell). It
provisions the environment, downloads the models, runs the correctness tests,
and then benchmarks. It is resumable: every stage skips itself if its output
artifact already exists.

Once the run finishes, `docs/results.md` is regenerated from the raw
measurements by `scripts/make_results_doc.py`. That file is machine-written; a
configuration that has not run shows `NOT MEASURED` rather than a number.

---

## Why speculative decoding is lossless

Speculative decoding does not approximate the target model's output distribution.
The accept/reject rule is constructed so that the marginal distribution of every
emitted token is **provably identical** to the target distribution. The draft
model affects only throughput — a better draft improves speed, a poor draft
falls back gracefully to standard sampling. Quality is unchanged.

The proof is in `tests/test_sampler_synthetic.py`: a Monte Carlo test over
synthetic distributions (1,000,000 trials, adversarial cases included) confirms
that the empirical output distribution matches the target distribution with
chi-square p > 0.001 and total variation distance within Monte Carlo error.

---

## Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti Mobile (TU117) |
| VRAM | 4 GB physical, ~3.4 GB usable |
| Compute capability | SM 7.5 (Turing, no FlashAttention-2) |
| FP16 | Yes (no bfloat16, no FP8) |
| Memory bandwidth | 192 GB/s |
| CPU | AMD Ryzen 7 4800H |
| System RAM | 16 GB |
| OS | Windows |

---

## Models

| Role | Model | Parameters | FP16 size |
|---|---|---|---|
| Target | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1 B | 2.20 GB |
| Draft (Phase 1) | `JackFram/llama-68m` | 68 M | 0.14 GB |
| Draft (Phase 2) | EAGLE head (trained offline) | ~60 M | 0.24 GB |

Both Phase 1 models share the Llama tokenizer (vocabulary size 32,000),
which is required by the accept/reject rule.

---

## Quickstart

### 1. Install dependencies

```powershell
# CUDA torch build (required — default pip install gives CPU-only)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt
```

### 2. Download models

```powershell
python scripts/download_models.py
```

Downloads `TinyLlama-1.1B-Chat-v1.0` and `llama-68m` to `models/`.
After this step, benchmarking runs fully offline.

### 3. Verify environment (Phase 0)

```powershell
python scripts/run_vram_probe.py
```

Asserts CUDA availability, SM 7.5 capability, FP16 support, and measures
actual VRAM usage. Results go to `results/vram_probe.json`.

### 4. Run correctness tests

```powershell
# Fast tests (CPU, no models needed):
pytest tests/test_sampler_synthetic.py -v

# Full suite (requires CUDA + downloaded models):
pytest tests/ -v -m "not slow"

# End-to-end distribution test (slow, ~20,000 generation passes):
pytest tests/test_distribution_match.py -v -m slow
```

### 5. Run Phase 1 benchmark

```powershell
# Single K value:
python scripts/run_phase1_bench.py --K 4

# Sweep K over {1, 2, 4, 8}:
python scripts/run_phase1_bench.py --sweep-K
```

Results are written to `results/raw_measurements.jsonl` and aggregated to
`results/summary.json`.

---

## Results Summary

| Configuration | Best K | Speedup | 95% CI | α (tokens/round) |
|---|---|---|---|---|
| Phase 1, sampling (T = 1.0) | 3 | 1.321x | 1.235–1.406 | 1.703 |
| Phase 1, greedy (T = 0.0) | 2 | 1.442x | 1.373–1.510 | 1.722 |
| Phase 2, EAGLE head | 1 | 0.938x | 0.921–0.956 | 1.074 |

Baselines: 11.40 tokens/s sampling, 11.47 tokens/s greedy, target model only.
See `docs/results.md` for the full sweep, latency percentiles and per-domain
breakdown — that file is machine-written from the raw measurements, and a
configuration that has not run shows `NOT MEASURED` rather than a number.

All reported speedups carry 95% bootstrap confidence intervals computed as
the ratio of aggregate throughputs (ratio-of-means, paired resampling over
prompts, B=10,000 resamples). Bare point estimates without intervals are not
reported.

---

## Repository Structure

```
specbench/
├── specbench/           Core library
│   ├── models.py        Model loading, VRAM probes
│   ├── sampling.py      Shared logits processors
│   ├── kv.py            KV cache and rollback
│   ├── spec_decode.py   Speculative decoding loop
│   ├── baseline.py      Target-only reference
│   ├── harness.py       Benchmark runner
│   ├── stats.py         Bootstrap CIs, chi-square, TVD
│   └── eagle/           Phase 2: EAGLE draft head
├── tests/               Correctness test suite
├── scripts/             Setup and benchmark runners
├── results/             Measurements (gitignored: hidden_states/)
└── docs/                Architecture, methodology, results, limitations
```

---

## Documentation

- [Architecture](docs/architecture.md) — component breakdown, data flow, KV rollback design
- [VRAM Budget](docs/vram-budget.md) — arithmetic, rejected configurations, measured values
- [Methodology](docs/methodology.md) — prompt set, warmup, interleaved A/B design, bootstrap CI
- [Results](docs/results.md) — all measurements with confidence intervals
- [Limitations](docs/limitations.md) — honest scope constraints and pending items

---

## References

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
