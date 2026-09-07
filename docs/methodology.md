# Measurement Methodology

## Prompt Set

The benchmark uses a fixed set of 30 prompts committed in `specbench/harness.py`.
The set covers two domains:

- **Prose** (16 prompts): historical, scientific, philosophical, and narrative topics.
- **Code** (10 prompts): Python algorithm stubs and class definitions.
- **Mixed / instructional** (6 prompts): technical explanations and engineering questions.

Domain separation matters: prose and code have different acceptance rate profiles
because language model predictions are more certain in syntactically constrained code
contexts. Reporting a single blended number without domain breakdown obscures this.

The prompt set slice is fixed and reproducible. No prompt is generated at runtime.

---

## Warmup Policy

The first 2 benchmark runs (one baseline, one speculative) per condition are
discarded. These runs are contaminated by:

- CUDA context creation
- PyTorch kernel autotuning
- GPU clock ramp-up from idle state

Warmup runs are still executed (to pay the startup cost properly) but not
included in reported statistics.

---

## Interleaved multi-arm design

This is the most important methodological decision in the harness.

Every condition is an **arm**: the target-only baseline, and one arm per
speculation depth (and, in Phase 2, per draft-head configuration). For each
prompt, every arm runs exactly once, back to back, and the order of the arms is
**rotated** by one position from prompt to prompt:

```
Prompt 0:  baseline, K=1, K=2, K=3, K=4, K=6, K=8
Prompt 1:  K=1, K=2, K=3, K=4, K=6, K=8, baseline
Prompt 2:  K=2, K=3, K=4, K=6, K=8, baseline, K=1
...
```

Three things follow from this.

**Thermal drift is shared, not attributed.** The alternative — all baseline
runs first, then all speculative runs — is common and wrong on thermally
limited hardware. A GTX 1650 Ti in a laptop chassis throttles from ~1485 MHz to
roughly 800–1000 MHz within the first five to ten minutes of sustained load. A
blocked design charges that entire drift to whichever condition ran second, and
the resulting bias goes in whichever direction happens to flatter the
experimenter. Interleaving puts every arm at essentially the same point on the
thermal ramp for a given prompt.

**Order within a prompt is not confounded with arm identity.** Interleaving
alone is not enough: whichever arm always runs first pays a slightly colder
cache and whichever runs last sits at the local temperature peak. Rotating the
order spreads that cost evenly. Each record stores `arm_position`, so the
residual effect can be estimated after the fact rather than assumed away.

**One baseline serves every arm.** Running a separate baseline per speculation
depth would mean six or seven measurements of the same quantity, taken at six
or seven different points in a thermal ramp — slower and noisier than measuring
it once per prompt and pairing every arm against it. Pairing is what makes the
bootstrap below valid.

GPU temperature and clock are recorded per run, so drift can be checked
post hoc rather than taken on trust.

---

## Bootstrap Confidence Intervals

Speedup is reported as the **ratio of aggregate throughputs** (tokens/sec),
not the mean of per-prompt ratios.

```
speedup = mean(spec_tps) / mean(baseline_tps)
```

The mean-of-ratios estimator is biased upward (Jensen's inequality applied to
the reciprocal) and its distribution is harder to characterise. The
ratio-of-means is the standard approach in systems benchmarking.

Bootstrap procedure:
1. For each resample (B = 10,000 resamples):
   a. Draw `n` prompt indices with replacement.
   b. Compute `mean(spec_tps[idx]) / mean(baseline_tps[idx])`.
2. Report the 2.5th and 97.5th percentiles as the 95% confidence interval.

Resampling is **paired**: the same prompt indices are drawn for both conditions.
This preserves the pairing and gives a tighter interval (prompt difficulty
affects both conditions in the same direction).

Results are presented in the form `<speedup>× (95% CI <low>–<high>)`. The
numbers in that sentence are a format description, not a measurement; actual
values live only in `docs/results.md`, which is generated from the raw data by
`scripts/make_results_doc.py` and never edited by hand.

A speedup without a confidence interval is not a complete result.

---

## Acceptance Length

Mean acceptance length α is defined as the mean number of tokens emitted per
target forward pass (accepted tokens + 1 for the resampled or bonus token).
This is the fundamental quantity; wall-clock speedup is bounded by α because
the target model cannot run faster than once per token in the worst case.

The relationship between α and speedup is sub-linear in practice because
each additional draft forward pass per round has a non-zero cost.

---

## Latency: what is measured, and what is not

Speculative decoding has characteristically **bursty** latency. A verification
pass emits several tokens at the same instant, then the system pauses for the
next round (K draft steps plus one verify). The distribution is bimodal: a
spike at 0 ms for tokens that arrived together, and a second mode at the round
duration. A mean hides exactly the property a user perceives as stutter.

Two series are therefore recorded separately, and neither is inferred from the
other:

| Series | Definition | Baseline meaning |
|---|---|---|
| `round_latencies_ms` | Wall-clock duration of one decoding round | Time per token |
| `token_arrival_gaps_ms` | Interval between consecutive tokens becoming available | Time per token |

For the speculative path, `token_arrival_gaps_ms` is 0.0 for every token after
the first in a round. That is not a rounding artefact: those tokens genuinely
became available simultaneously, out of one verification pass. p50 of that
series can legitimately be 0 ms while p90 is large, and that bimodality is the
finding, not a defect in the measurement.

### How the timestamps are taken

The generators themselves record a timestamp at the end of each round, and the
harness derives both series from those. Timestamps are placed immediately after
a `.item()` call that has already forced a device synchronisation, so no extra
`cuda.synchronize()` is inserted anywhere inside the measured region. This
matters for fairness: the baseline synchronises once per token and the
speculative path once per round, so adding explicit barriers per timestamp
would tax the baseline more heavily and manufacture a speedup.

### What this replaces

An earlier version of the harness recorded `wall_clock / n_tokens` repeated
`n_tokens` times. Every percentile of that series is the mean by construction:
p50 = p90 = p99, standard deviation exactly 0. Percentiles computed from a mean
are not measurements of anything, and the burstiness above — the entire reason
to report percentiles for this algorithm — is invisible in them.

---

## GPU Monitoring

GPU temperature and clock frequency are recorded per run using:

```
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics --format=csv,noheader,nounits
```

These values are stored in `results/raw_measurements.jsonl` alongside each run.
If thermal throttling was significant during a run, this can be detected
post-hoc by plotting clock vs. time.

---

## Hardware and Software Provenance

The following information is recorded in every result row:

| Field | Value |
|---|---|
| `torch_version` | `2.5.1+cu121` |
| `cuda_version` | `12.1` |
| `driver_version` | _recorded at runtime_ |
| `device` | `NVIDIA GeForce GTX 1650 Ti` |
| `device_capability` | `[7, 5]` |

### Resolved Package Versions

Installed via:
```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

| Package | Pinned | Resolved |
|---|---|---|
| torch | 2.5.1 | _verify with `pip show torch`_ |
| transformers | 4.46.3 | _verify_ |
| accelerate | 1.1.1 | _verify_ |
| safetensors | 0.4.5 | _verify_ |
| huggingface-hub | 0.26.2 | _verify_ |
| numpy | 1.26.4 | _verify_ |
| scipy | 1.14.1 | _verify_ |
| pandas | 2.2.3 | _verify_ |
| matplotlib | 3.9.2 | _verify_ |
| pytest | 8.3.3 | _verify_ |
| tqdm | 4.67.0 | _verify_ |

Run `pip freeze > installed.txt` after setup and commit `installed.txt`
to record the actual resolved versions.
