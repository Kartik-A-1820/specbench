# VRAM Budget

## Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti Mobile (TU117) |
| Physical VRAM | 4 GB |
| Usable VRAM | ~3.4 GB (after WDDM overhead) |
| Compute capability | SM 7.5 (Turing) |
| Memory bandwidth | 192 GB/s |

---

## Selected Configuration

| Role | Model | Parameters | FP16 weight size |
|---|---|---|---|
| Target | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.10 B | 2.20 GB |
| Draft (Phase 1) | `JackFram/llama-68m` | 68 M | 0.14 GB |

Both models use the Llama tokenizer (vocabulary size 32,000). Shared vocabulary
is required by the accept/reject rule.

---

## KV Cache Arithmetic

**TinyLlama-1.1B:**

Architecture: 22 layers, 4 KV heads (grouped-query attention), head_dim 64.

```
Bytes per token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes
               = 2 × 22 × 4 × 64 × 2
               = 22,528 bytes ≈ 22 KB/token
→ 46 MB at 2048 context
```

**llama-68m:**

Architecture: 2 layers, 12 heads (multi-head, no GQA), head_dim 64.

```
Bytes per token = 2 × 2 × 12 × 64 × 2
               = 6,144 bytes ≈ 6 KB/token
→ 13 MB at 2048 context
```

TinyLlama's grouped-query attention (4 KV heads vs. 32 query heads) is what
makes the KV cache 8× smaller per token compared to a full-MHA model of the
same width. This is the primary reason the configuration fits.

---

## Predicted Budget at 2048 Context

| Component | Predicted |
|---|---|
| Target weights (FP16) | 2.20 GB |
| Draft weights (FP16) | 0.14 GB |
| Target KV cache | 0.05 GB |
| Draft KV cache | 0.01 GB |
| CUDA context + cuBLAS | ~0.30 GB |
| Transient activations | ~0.10 GB |
| **Total** | **~2.80 GB** |

Predicted headroom against 3.4 GB: ~0.60 GB.

---

## Measured Values (Phase 0)

> **Note:** The table below will be filled in after running `scripts/run_vram_probe.py`.
> The raw numbers will be in `results/vram_probe.json`.

| Component | Predicted | Measured | Discrepancy |
|---|---|---|---|
| CUDA context | ~0.30 GB | _TBD_ | — |
| Target weights | 2.20 GB | _TBD_ | — |
| Draft weights | 0.14 GB | _TBD_ | — |
| Target KV cache (2048 tok) | 0.05 GB | _TBD_ | — |
| Draft KV cache (2048 tok) | 0.01 GB | _TBD_ | — |
| **Peak total** | **2.80 GB** | **_TBD_** | — |

To fill in the measured values:

```powershell
python scripts/run_vram_probe.py
```

---

## Rejected Configurations

These were considered and rejected because they exceed the 3.4 GB budget.

### Qwen2.5-1.5B-Instruct + Qwen2.5-0.5B-Instruct

The natural same-family pair with good acceptance rates.

```
Target weights: 1.54 B × 2 bytes = 3.09 GB
Draft weights:  0.494 B × 2 bytes = 0.99 GB
Total weights alone: 4.08 GB > 3.4 GB
```

Rejected before KV cache is even allocated.

### Pythia-1.4B + Pythia-160M

The classic research pair for speculative decoding papers.

```
Target weights: 2.83 GB
Draft weights:  0.32 GB
Subtotal:       3.15 GB
```

Pythia uses full multi-head attention with `head_dim = 128`:

```
Target KV/token = 2 × 24 × 16 × 128 × 2 = 196,608 B ≈ 192 KB/token
At 1024 context: 0.197 GB for target KV alone
Total at 1024:   3.15 + 0.197 ≈ 3.35 GB
```

This only just fits at 1024 context (a very short generation window) and
leaves essentially no room for Phase 2 training state. Rejected.
