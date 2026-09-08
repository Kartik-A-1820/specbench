"""
specbench/costmodel.py — the two-parameter round-cost model, in one place.

The model
---------
A speculative round costs one target forward pass plus K draft passes. Writing
c for the cost of a draft pass in units of a target pass, and alpha for the
mean number of tokens the round emits:

    time per token  =  (1 + K * c) / alpha    target-pass-equivalents
    speedup         =  alpha / (1 + K * c)

alpha is measured directly — it is the mean of tokens_per_round, which the
generator records. c is not directly observable from a throughput number, so
it is recovered two different ways, and the two ways are worth keeping
separate:

  fit_c_from_arms()
      Solves for c from the benchmark itself. Total time for an arm is
      n_target * t_target + n_draft * t_draft, with t_target taken from the
      baseline arm's own throughput. Every quantity is measured; nothing is
      assumed. This is the post-hoc fit.

  c_from_direct_timing()
      Times a single forward pass of each model in isolation. This can be run
      before the sweep, which makes it possible to state a prediction in
      advance and then check it, rather than fitting a curve to data and
      declaring that the curve fits.

Both were originally written inline inside scripts/build_kt_docx.py, which made
them hard to reuse and impossible to test on their own. They are here so that
the document builder, and anything that later wants to apply the same
arithmetic to a second model pair, run literally the same code — "the same cost
model" is only a meaningful claim if it is one implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def predicted_speedup(alpha: float, K: int, c: float) -> float:
    """speedup = alpha / (1 + K*c)."""
    return alpha / (1.0 + K * c)


def optimal_k(alpha_of_k, k_values, c: float):
    """
    The K in `k_values` maximising alpha(K) / (1 + K*c).

    Takes alpha as a function of K rather than a constant, because it is not
    one: acceptance length grows with K but sub-linearly, which is the entire
    reason the sweep has an interior optimum.
    """
    best_k, best = None, -1.0
    for K in k_values:
        s = predicted_speedup(alpha_of_k(K), K, c)
        if s > best:
            best, best_k = s, K
    return best_k, best


def target_pass_ms(baseline: dict) -> Optional[float]:
    """
    Cost of one target forward pass in ms, from the baseline arm's throughput.

    The baseline emits max_new_tokens tokens using mean_target_forward_calls
    passes, so the division is exact rather than a model of anything.
    """
    tps = baseline.get("mean_tps") or 0.0
    calls = baseline.get("mean_target_forward_calls") or 0.0
    n_tokens = baseline.get("max_new_tokens", 128)
    if tps <= 0 or calls <= 0:
        return None
    return (n_tokens / tps) / calls * 1000.0


def fit_c_from_arms(baseline: dict, spec_by_k: Dict[int, dict]) -> dict:
    """
    Recover c from a measured sweep.

    For each speculative arm:
        total_time = n_target * t_target + n_draft * t_draft
    with t_target known from the baseline, this is one equation in one
    unknown. Solving it per arm and comparing the answers is a consistency
    check in itself: if c came out wildly different at K=1 and K=6, the
    model's assumption that a draft pass costs the same regardless of K would
    be wrong.

    Returns a dict with per-K estimates, the mean c, and the predicted vs.
    measured speedup at every K.
    """
    t_target_ms = target_pass_ms(baseline)
    if t_target_ms is None or not spec_by_k:
        return {"ok": False, "reason": "no baseline throughput or no arms"}

    n_tokens = baseline.get("max_new_tokens", 128)
    per_k = []
    for K in sorted(spec_by_k):
        s = spec_by_k[K]
        tps = s.get("mean_tps") or 0.0
        n_t = s.get("mean_target_forward_calls") or 0.0
        n_d = s.get("mean_draft_forward_calls") or 0.0
        if tps <= 0 or n_d <= 0:
            continue
        total_ms = n_tokens / tps * 1000.0
        t_draft_ms = (total_ms - n_t * t_target_ms) / n_d
        per_k.append({
            "K": K,
            "t_draft_ms": t_draft_ms,
            "c": t_draft_ms / t_target_ms,
            "alpha": s.get("mean_acceptance_length", 0.0),
            "measured_speedup": s.get("speedup", 0.0),
        })

    if not per_k:
        return {"ok": False, "reason": "no usable speculative arms"}

    c_mean = float(np.mean([e["c"] for e in per_k]))
    c_spread = float(np.max([e["c"] for e in per_k]) - np.min([e["c"] for e in per_k]))

    for e in per_k:
        e["predicted_speedup"] = predicted_speedup(e["alpha"], e["K"], c_mean)
        m = e["measured_speedup"]
        e["error_pct"] = (
            (e["predicted_speedup"] - m) / m * 100.0 if m else float("nan")
        )

    errors = [abs(e["error_pct"]) for e in per_k if np.isfinite(e["error_pct"])]
    return {
        "ok": True,
        "t_target_ms": t_target_ms,
        "c": c_mean,
        "c_spread": c_spread,
        "per_k": per_k,
        "mean_abs_error_pct": float(np.mean(errors)) if errors else float("nan"),
        "max_abs_error_pct": float(np.max(errors)) if errors else float("nan"),
        "baseline_tps": baseline.get("mean_tps"),
    }


def c_from_direct_timing(t_target_ms: float, t_draft_ms: float) -> float:
    """c as the straightforward ratio of two independently timed forward passes."""
    return t_draft_ms / t_target_ms


def alpha_needed_for_breakeven(K: int, c: float) -> float:
    """
    The acceptance length at which speculation stops losing.

    Speedup exceeds 1 exactly when alpha > 1 + K*c. Quoting this alongside a
    measured alpha turns "it was slower" into "it needed 2.3 tokens per round
    and got 1.6", which is a diagnosis rather than an outcome.
    """
    return 1.0 + K * c
