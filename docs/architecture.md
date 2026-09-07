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
  pending token p  (emitted last round; deliberately NOT in either KV cache)
         │
         ▼
   draft model                      target model
   ┌────────────┐                   ┌──────────────────────┐
   │ K x 1-token│  x_1 .. x_K       │ ONE forward pass over│
   │ autoregress│ ────────────────► │  [p, x_1, ..., x_K]  │
   │  + q_1..q_K│                   │      (K+1 tokens)    │
   └────────────┘                   └──────────────────────┘
                                              │
                       row 0 -> p_1 ...  row K -> p_{K+1}
                                              │
                                              ▼
                                   accept / reject, in order
                              ┌───────────────────────────────┐
                              │ for i = 1..K:                 │
                              │   r ~ Uniform(0,1)            │
                              │   if r < min(1, p_i/q_i):     │
                              │       accept x_i              │
                              │   else:                       │
                              │       emit ~ norm(max(0,p-q)) │
                              │       stop                    │
                              └───────────────────────────────┘
                                              │
                          all K accepted -> bonus token from p_{K+1}
                                              │
                                              ▼
                       roll both caches to committed_len + 1 + accepted
                       last emitted token becomes the next `pending`
```

### Why the block is K+1 tokens, not K

The accept/reject rule needs K+1 target distributions per round: p_1..p_K to
judge the proposals, and p_{K+1} for the bonus token. A K-token forward pass
yields only K logit rows, and the one it is missing is p_1 — the distribution
over the first proposal, conditioned on the prefix alone, which is determined
before any proposal has been fed.

The obvious fix is to spend a second, single-token target forward pass per
round to obtain it. That is also the expensive fix: the target pass is the
dominant cost of a round, so paying for two of them caps the achievable speedup
at roughly half the acceptance length, and for realistic acceptance rates the
implementation measures *slower* than the baseline it is meant to beat.

Instead the loop carries a **pending** token: the token emitted at the end of
the previous round, committed to the output but held out of both KV caches. It
heads the next round's block. Feeding `[pending, x_1..x_K]` makes output row 0
exactly p_1, row i exactly p_{i+1}, and row K exactly p_{K+1} — all K+1
distributions, from one pass.

Cache bookkeeping follows directly. Entering a round both caches hold
`committed_len` positions and exclude `pending`; after accepting `a`
proposals the committed prefix is `committed_len + 1 + a`, where the `+1` is
`pending` itself, which is correct by construction because it was emitted
rather than proposed. The draft only needs `pending` plus `x_1..x_{K-1}` to
produce all K proposals, so it runs one position short and is topped up with a
single extra draft call in the one case where that matters, `a == K`.

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

At inference time the head autoregresses on the target's own last-layer
features rather than on tokens. The pairing matters and is easy to get wrong.
Writing `f_t` for the target feature at position `t`, the head is trained on

```
head(f_{t-1}, embed(token_t))  ->  f_t
```

and the target's frozen LM head maps `f_t` to the distribution over token
`t+1`. Inference must use the same pairing. Feeding `(f_t, embed(token_t))`
instead — one position later than anything seen in training — costs no
correctness, because the accept/reject rule still guards the output, but it
destroys the acceptance rate, so the failure shows up as "EAGLE is no faster
than the baseline" rather than as a crash.

```
For each draft step:
  head(feature, embed(current_token)) -> predicted_feature
  lm_head_frozen(predicted_feature)   -> q over the next token
  sample x ~ q
  feature       = predicted_feature      # head's own output from here on
  current_token = x
```

The first step of a round uses the *real* target feature; every subsequent
step uses the head's prediction. That is the EAGLE formulation, and it is why
the head has to be able to run on its own output distribution without drifting.

The verification pass is the same K+1-token block as Phase 1, requested with
`output_hidden_states=True`. Block index `j` carries the real feature at
absolute position `committed_len + j`, so after accepting `a` proposals the
feature needed to seed the next round — the predecessor of the newly emitted
token — is block index `a`, already in hand. The EAGLE path therefore also
needs exactly one target forward pass per round.

The accept/reject sampler is reused from `spec_decode.py` unchanged. Only the
source of `q` changes, so the distribution guarantee carries over identically:
a badly trained head costs speed, never correctness.

---

## Why SDPA, Not FlashAttention-2

FlashAttention-2 requires SM 8.0 (Ampere) or higher. The GTX 1650 Ti is
SM 7.5 (Turing). PyTorch SDPA (`attn_implementation="sdpa"`) is used instead;
it is available on SM 7.5 and handles FP16 correctly. An assertion in
`models.verify_cuda_environment()` prevents accidental use on a wrong device.
