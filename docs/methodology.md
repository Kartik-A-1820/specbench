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

## Interleaved A/B Design

Baseline and speculative runs **alternate** prompt-by-prompt:

```
Prompt 0: baseline → spec
Prompt 1: baseline → spec
...
Prompt 29: baseline → spec
```

This is the most important methodological decision in the harness.

The alternative — running all baseline conditions first, then all speculative
conditions — is common but incorrect for thermally limited hardware. A GTX 1650
Ti in sustained operation will throttle its clock from ~1485 MHz to ~800–1000 MHz
within the first 5–10 minutes of load. A blocked design attributes this drift to
whichever condition ran second, inflating the speedup estimate by an amount
proportional to how much the clock dropped.

The interleaved design ensures both conditions see the same thermal state at
every prompt. GPU temperature and clock are recorded per run so thermal drift
can be checked after the fact.

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

Results are presented as: `1.47× (95% CI 1.38–1.56)`.
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

## Latency Distribution

Per-token latency is reported as p50, p90, p99 of inter-token delays.

Speculative decoding has characteristically **bursty** latency:
- A verify pass emits several tokens at once (near-zero inter-token delay
  for the batch of accepted tokens).
- Then the model pauses to complete the next round (draft K steps + verify).

The distribution is bimodal: a sharp peak near 0 ms (tokens from a successful
verify batch) and a peak at the verify-pass latency. The mean hides this
bimodality, which is what the user actually perceives as a "stutter" pattern.
Reporting percentiles makes the bimodality visible.

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
