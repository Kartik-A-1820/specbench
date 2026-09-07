# Architecture

## Overview

specbench implements vanilla speculative decoding (Phase 1) and an
EAGLE-style trained draft head (Phase 2) for the `TinyLlama-1.1B-Chat-v1.0`
target model running on a GTX 1650 Ti Mobile (SM 7.5, 3.4 GB usable VRAM).

---

## Component Breakdown

```
specbench/
├── models.py        Hardware assertions, FP16/SDPA model loading, VRAM probes.
├── sampling.py      Temperature / top-k / top-p logits processors (shared path).
├── kv.py            KV cache helpers: prefill, single/multi-token forward, rollback.
├── spec_decode.py   Core speculative decoding loop (Leviathan et al. 2023).
├── baseline.py      Target-only autoregressive reference.
├── harness.py       Timing, interleaved A/B runner, raw_measurements.jsonl writer.
├── stats.py         Bootstrap CIs, chi-square tests, TVD, latency percentiles.
└── eagle/
    ├── head.py             EagleDraftHead module.
    ├── collect.py          Offline hidden-state collection.
    ├── train.py            Head training loop (target model unloaded).
    └── spec_decode_eagle.py  EAGLE speculative decoding loop.
```

---

## Data Flow: One Speculative Round

```
                 draft model
                 ┌──────────┐
Last accepted ──►│ K×autoReg├──► proposed tokens x₁…xₖ
token             │          │    + draft dists q₁…qₖ
                 └──────────┘
                       │
                       ▼
                 target model
                 ┌──────────┐
prefix + x₁…xₖ ►│ 1 forward├──► target dists p₁…p_{K+1}
                 └──────────┘
                       │
                       ▼
              accept/reject rule
              ┌──────────────────┐
              │ For each i=1..K: │
              │  r ~ Uniform(0,1)│
              │  if r < p_i/q_i  │──► accept xᵢ
              │  else            │──► sample residual; stop
              └──────────────────┘
                       │
              If all K accepted:
                bonus token from p_{K+1}
                       │
                       ▼
              Roll back both KV caches
              to accepted prefix length
```

---

## KV Cache Rollback Design

Both models maintain `DynamicCache` objects whose key/value tensors grow with
each forward pass. After a rejection at position `i` (0-indexed), `i-1` tokens
have been accepted. The draft cache holds `K` speculative entries beyond the
prefix; the target cache holds `K` entries from the verification pass. Both
are truncated to `prefix_len + (i-1)` via `specbench/kv.rollback_cache()`,
which slices the `seq_len` dimension of every layer's K and V tensors.

Getting this wrong is the most common source of corrupted-but-plausible output
in speculative decoding implementations. `tests/test_kv_rollback.py` covers
this by comparing generation from a rolled-back cache against a freshly built
cache over the same prefix.

---

## Logits Processor Symmetry

The same `apply_logits_processors()` call from `specbench/sampling.py` is used
for both the draft distribution `q` and the target distribution `p` before
computing the acceptance ratio and residual. Applying temperature, top-k, or
top-p asymmetrically would break the distribution guarantee. The shared
function is the enforcement mechanism.

---

## EAGLE Offline-Caching Decision

Training the EAGLE head with the target model resident in VRAM does not fit
within the 3.4 GB budget:

| Component                    | Size     |
|------------------------------|----------|
| Target weights (frozen, FP16) | 2.20 GB  |
| Head params (~60 M, FP32)     | 0.24 GB  |
| Gradients                     | 0.24 GB  |
| AdamW m + v states            | 0.48 GB  |
| Activations                   | ~0.30 GB |
| **Total**                     | **~3.5 GB > 3.4 GB** |

The solution is to run the target over the training corpus once, write
last-layer hidden states to disk in FP16 (~4 KB/token), and then unload the
target before training begins. With the target absent, the training budget is:

| Component                    | Size     |
|------------------------------|----------|
| Head + gradients + AdamW     | ~0.96 GB |
| Frozen embedding + LM head   | ~0.13 GB |
| Activations (batch 4, seq 64) | ~0.20 GB |
| **Total**                    | **~1.29 GB** |

This is a genuine engineering decision forced by the hardware, not an
optimisation. It is reflected in the two-step workflow of `collect.py` then
`train.py`.

---

## EAGLE Inference Architecture

At inference time, the head autoregresses on hidden states rather than tokens:

```
For each draft step:
  head(target_hidden_t, embed(token_t)) → predicted_hidden_{t+1}
  lm_head_frozen(predicted_hidden_{t+1}) → q_{t+1} (logits)
  sample x_{t+1} ~ q_{t+1}
  token_{t+1} = x_{t+1}
  hidden_{t+1} = predicted_hidden_{t+1}  (for next step)
```

After K steps, the target verifies all K proposals in one forward pass,
returning its own hidden states which seed the next round.

The accept/reject sampler is reused from `spec_decode.py` unchanged. Only
the source of `q` changes. The distribution guarantee carries over identically.

---

## Why SDPA, Not FlashAttention-2

FlashAttention-2 requires SM 8.0 (Ampere) or higher. The GTX 1650 Ti is
SM 7.5 (Turing). PyTorch SDPA (`attn_implementation="sdpa"`) is used instead;
it is available on SM 7.5 and handles FP16 correctly. An assertion in
`models.verify_cuda_environment()` prevents accidental use on a wrong device.
