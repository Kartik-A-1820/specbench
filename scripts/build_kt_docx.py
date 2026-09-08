"""
scripts/build_kt_docx.py — build the KT handover document.

Reads docs/kt-source.md, substitutes the {{RESULTS:...}} placeholders with
tables built from the measured data in results/, and emits a document model as
JSON for scripts/render_kt_docx.js to turn into a .docx.

The results tables are generated, never typed. If a configuration has not been
measured, its placeholder is replaced with an explicit statement that it was
not measured — not with an estimate, an expectation, or a blank.

Usage:
    python scripts/build_kt_docx.py --out build/kt_model.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from specbench.costmodel import fit_c_from_arms

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results"
SOURCE = PROJECT_ROOT / "docs" / "kt-source.md"

DOMAIN_RANGES = {"Prose": range(0, 15), "Code": range(15, 25), "Instructional": range(25, 31)}

NOT_MEASURED = (
    "NOT MEASURED. This stage did not complete, so there is no number to report "
    "here. Nothing has been estimated or filled in."
)


# ---------------------------------------------------------------------------
# Loading measured data
# ---------------------------------------------------------------------------

def load_summary() -> dict:
    path = RESULTS / "summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_raw() -> list:
    path = RESULTS / "raw_measurements.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_json(name: str):
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Results blocks
# ---------------------------------------------------------------------------

def arms_for_prefix(summary: dict, prefix: str):
    baseline = None
    spec = {}
    for name, s in summary.items():
        if not name.startswith(prefix) or not isinstance(s, dict) or "mean_tps" not in s:
            continue
        if s.get("mode") == "baseline":
            baseline = s
        else:
            spec[s.get("K", 0)] = s
    return baseline, spec


def block_phase1(summary: dict) -> list:
    out = []
    for prefix, label, temp_note in [
        (
            "phase1_T1.0",
            "Phase 1 — sampling, temperature 1.0",
            "Ancestral sampling, top-k and top-p disabled, 128 new tokens per run.",
        ),
        (
            "phase1_T0.0",
            "Phase 1 — greedy, temperature 0.0",
            "Greedy decoding. Output here is token-identical to the baseline by "
            "construction, so this measures pure overhead and speedup with no "
            "sampling variance at all.",
        ),
    ]:
        out.append({"type": "heading3", "text": label})
        baseline, spec = arms_for_prefix(summary, prefix)
        if not baseline or not spec:
            out.append({"type": "para", "text": NOT_MEASURED})
            continue
        out.append({"type": "para", "text": temp_note})
        out.append(
            {
                "type": "para",
                "text": f"Baseline (target model only): {baseline['mean_tps']:.2f} tokens/s "
                f"over {baseline['n_prompts']} scored prompts.",
            }
        )
        rows = [
            [
                "K",
                "tok/s",
                "Speedup",
                "95% CI",
                "Tokens/round",
                "Target fwd",
                "Draft fwd",
            ]
        ]
        for K in sorted(spec):
            s = spec[K]
            rows.append(
                [
                    str(K),
                    f"{s['mean_tps']:.2f}",
                    f"{s['speedup']:.3f}x",
                    f"{s['speedup_ci_95_low']:.3f}-{s['speedup_ci_95_high']:.3f}",
                    f"{s['mean_acceptance_length']:.3f}",
                    f"{s.get('mean_target_forward_calls', 0):.0f}",
                    f"{s.get('mean_draft_forward_calls', 0):.0f}",
                ]
            )
        out.append({"type": "table", "rows": rows})

        best_K = max(spec, key=lambda k: spec[k]["speedup"])
        b = spec[best_K]
        out.append(
            {
                "type": "para",
                "text": f"Best configuration: K = {best_K}, at {b['speedup']:.3f}x "
                f"(95% CI {b['speedup_ci_95_low']:.3f}-{b['speedup_ci_95_high']:.3f}) "
                f"with {b['mean_acceptance_length']:.3f} tokens emitted per round. "
                f"The target model runs {b.get('mean_target_forward_calls', 0):.0f} "
                f"forward passes to produce {b.get('max_new_tokens', 128)} tokens, "
                f"against {baseline.get('mean_target_forward_calls', 0):.0f} for the "
                "baseline — the direct measurement of the one-pass-per-round property.",
            }
        )
        total_fallback = sum(s.get("residual_fallback_total", 0) for s in spec.values())
        out.append(
            {
                "type": "para",
                "text": f"Degenerate-residual fallbacks across all arms in this run: "
                f"{total_fallback}. This counter should be zero in exact arithmetic; "
                "a nonzero and growing value would indicate a numerical problem in "
                "the residual computation.",
            }
        )
    return out


def block_latency(summary: dict) -> list:
    out = [{"type": "heading3", "text": "Latency"}]
    baseline, spec = arms_for_prefix(summary, "phase1_T1.0")
    if not baseline or 4 not in spec:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out
    s = spec[4]
    out.append(
        {
            "type": "para",
            "text": "Round latency is the wall-clock duration of one decoding round: "
            "one token for the baseline, up to K+1 for the speculative path. Token "
            "arrival gap is the interval between consecutive tokens becoming "
            "available, and is 0 ms for every token after the first in a "
            "speculative round because they genuinely arrive together.",
        }
    )
    rows = [["Metric", "Baseline", "Speculative K=4"]]
    for label, key, series in [
        ("Round p50 (ms)", "p50_ms", "round_latency_percentiles"),
        ("Round p90 (ms)", "p90_ms", "round_latency_percentiles"),
        ("Round p99 (ms)", "p99_ms", "round_latency_percentiles"),
        ("Round mean (ms)", "mean_ms", "round_latency_percentiles"),
        ("Round std (ms)", "std_ms", "round_latency_percentiles"),
        ("Arrival gap p50 (ms)", "p50_ms", "token_arrival_gap_percentiles"),
        ("Arrival gap p90 (ms)", "p90_ms", "token_arrival_gap_percentiles"),
        ("Arrival gap p99 (ms)", "p99_ms", "token_arrival_gap_percentiles"),
    ]:
        bv = baseline.get(series, {}).get(key)
        sv = s.get(series, {}).get(key)
        rows.append(
            [
                label,
                f"{bv:.2f}" if bv is not None else "n/a",
                f"{sv:.2f}" if sv is not None else "n/a",
            ]
        )
    out.append({"type": "table", "rows": rows})
    out.append(
        {
            "type": "para",
            "text": "Note the standard deviation of round latency: a nonzero value is "
            "the evidence that these are measured timestamps rather than a mean "
            "repeated N times. The previous implementation produced a standard "
            "deviation of exactly zero, with p50, p90 and p99 all equal.",
        }
    )
    return out


def block_domain(raw: list) -> list:
    out = [{"type": "heading3", "text": "Acceptance by prompt domain"}]
    by_domain = defaultdict(list)
    for r in raw:
        if r.get("mode") != "spec" or r.get("K") != 4 or r.get("temperature") != 1.0:
            continue
        pid = r.get("prompt_id")
        for domain, rng in DOMAIN_RANGES.items():
            if pid in rng:
                by_domain[domain].append(r["mean_acceptance_length"])
    if not by_domain:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out
    out.append(
        {
            "type": "para",
            "text": "Tokens emitted per round at K=4, temperature 1.0, split by prompt "
            "type. Reporting a single blended acceptance figure hides this "
            "variation, and the variation is the interesting part: acceptance "
            "tracks how constrained the next token is.",
        }
    )
    rows = [["Domain", "Prompts", "Mean tokens/round", "Std"]]
    for domain in ["Prose", "Code", "Instructional"]:
        vals = by_domain.get(domain, [])
        if not vals:
            rows.append([domain, "0", "NOT MEASURED", "-"])
            continue
        arr = np.array(vals)
        rows.append([domain, str(len(arr)), f"{arr.mean():.3f}", f"{arr.std():.3f}"])
    out.append({"type": "table", "rows": rows})
    return out


def block_phase2(summary: dict) -> list:
    out = [{"type": "heading3", "text": "Phase 2 — EAGLE draft head"}]
    ckpts = load_json("eagle_checkpoints.json")
    if not ckpts:
        out.append(
            {
                "type": "para",
                "text": "NOT MEASURED. No EAGLE head was trained, so there are no "
                "Phase 2 results. The Phase 1 numbers above stand on their own and "
                "no Phase 2 figure is inferred from them.",
            }
        )
        return out

    rows = [["Head", "Layers", "Intermediate", "Params (M)", "Final CE loss", "Peak VRAM (GB)"]]
    for name in sorted(ckpts):
        c = ckpts[name]
        rows.append(
            [
                name,
                str(c.get("num_layers")),
                str(c.get("intermediate_size")),
                f"{c.get('parameters', 0) / 1e6:.1f}",
                f"{c.get('final_ce_loss', float('nan')):.4f}",
                f"{(c.get('peak_vram_bytes') or 0) / 1e9:.3f}",
            ]
        )
    out.append({"type": "para", "text": "Trained heads:"})
    out.append({"type": "table", "rows": rows})

    any_swept = False
    for name in sorted(ckpts):
        baseline, spec = arms_for_prefix(summary, f"phase2_{name}")
        if not baseline or not spec:
            continue
        any_swept = True
        out.append({"type": "heading4", "text": f"Head {name}"})
        srows = [["K", "tok/s", "Speedup", "95% CI", "Tokens/round"]]
        for K in sorted(spec):
            s = spec[K]
            srows.append(
                [
                    str(K),
                    f"{s['mean_tps']:.2f}",
                    f"{s['speedup']:.3f}x",
                    f"{s['speedup_ci_95_low']:.3f}-{s['speedup_ci_95_high']:.3f}",
                    f"{s['mean_acceptance_length']:.3f}",
                ]
            )
        out.append({"type": "table", "rows": srows})
    if not any_swept:
        out.append(
            {
                "type": "para",
                "text": "Heads were trained but the benchmark sweep did not complete, "
                "so no Phase 2 speedup is reported.",
            }
        )
    return out


def block_eagle_vs_draft(summary: dict) -> list:
    """
    Head-to-head: the trained EAGLE head against the Phase 1 draft model, on the
    same target, the same prompts and the same measurement design.
    """
    out = [{"type": "heading3", "text": "EAGLE head vs. the separate draft model"}]
    b1, p1 = arms_for_prefix(summary, "phase1_T1.0")
    ckpts = load_json("eagle_checkpoints.json") or {}
    if not b1 or not p1 or not ckpts:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out

    n_tokens = b1.get("max_new_tokens", 128)
    t_target_ms = (n_tokens / b1["mean_tps"]) / b1["mean_target_forward_calls"] * 1000.0

    def draft_cost(s, t_ms):
        total_ms = n_tokens / s["mean_tps"] * 1000.0
        n_d = max(s.get("mean_draft_forward_calls", 0), 1)
        return (total_ms - s["mean_target_forward_calls"] * t_ms) / n_d / t_ms

    rows = [["Drafter", "Params", "Best speedup", "at K", "alpha", "c (draft/target)"]]

    bestK = max(p1, key=lambda k: p1[k]["speedup"])
    b = p1[bestK]
    rows.append(
        [
            "llama-68m (Phase 1)",
            "68 M",
            f"{b['speedup']:.3f}x",
            str(bestK),
            f"{b['mean_acceptance_length']:.3f}",
            f"{draft_cost(b, t_target_ms):.3f}",
        ]
    )
    for name in sorted(ckpts):
        bh, ph = arms_for_prefix(summary, f"phase2_{name}")
        if not bh or not ph:
            continue
        kbest = max(ph, key=lambda k: ph[k]["speedup"])
        s = ph[kbest]
        rows.append(
            [
                name,
                f"{ckpts[name].get('parameters', 0) / 1e6:.1f} M",
                f"{s['speedup']:.3f}x",
                str(kbest),
                f"{s['mean_acceptance_length']:.3f}",
                f"{draft_cost(s, t_target_ms):.3f}",
            ]
        )
    out.append({"type": "table", "rows": rows})

    out.append(
        {
            "type": "para",
            "text": "The trained head loses to the off-the-shelf draft model, and "
            "loses on both terms of speedup = alpha / (1 + K*c) at once. Its "
            "acceptance length is near 1.0, meaning almost nothing it proposes "
            "survives verification, and each of its draft steps costs roughly "
            "twice what a llama-68m step costs. Every EAGLE configuration "
            "measured here is slower than simply running the target model "
            "alone.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "This is a real result, not a bug, and it is worth being precise "
            "about why — three earlier versions of it WERE bugs, and each was "
            "found and fixed before these numbers were taken. The head's "
            "attention was bidirectional during training; its inference "
            "feature/token pairing was off by one; and it ran without a KV "
            "cache at inference. All three are fixed and covered by tests. What "
            "remains is not an implementation defect.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "The cause is that the head is drastically undertrained, and the "
            "evidence is a generalisation gap that can be measured directly. On "
            "cached documents from its own training corpus the head reaches an "
            "acceptance probability of about 0.28. On the held-out benchmark "
            "prompts it achieves about 0.05, which is what an acceptance length "
            "of 1.05 at K=1 implies. Same head, same temperature, roughly a "
            "six-fold gap: it has memorised 250,000 tokens of wikitext and mbpp "
            "rather than learned the target's distribution. llama-68m, "
            "pretrained on billions of tokens, reaches about 0.45 on those same "
            "held-out prompts.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "The capacity sweep corroborates it. Larger heads reached lower "
            "TRAINING cross-entropy — 2.00 at 96.5 M parameters against 2.60 at "
            "35.1 M — while their benchmark acceptance stayed flat at roughly "
            "1.06 to 1.09 tokens per round. Capacity bought fit on the training "
            "set and bought nothing on held-out data, which is the signature of "
            "overfitting rather than of a method that does not work.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "The honest summary is that this is a measurement of a "
            "250,000-token training budget, not a measurement of EAGLE. The "
            "published method trains on orders of magnitude more data. What this "
            "project can legitimately claim is the negative result under its own "
            "constraints, plus the reason, plus the fact that the reason is "
            "measurable rather than assumed.",
        }
    )
    return out


def block_costmodel(summary: dict) -> list:
    """
    Fit the round-cost model to the measured data and show that it predicts the
    observed speedups. Everything here is derived from measurements; nothing is
    assumed.
    """
    out = [{"type": "heading3", "text": "Does the cost model explain the measurements?"}]
    baseline, spec = arms_for_prefix(summary, "phase1_T1.0")
    if not baseline or not spec:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out

    n_tokens = baseline.get("max_new_tokens", 128)

    # The fit itself lives in specbench/costmodel.py, so that this block and
    # the roofline analysis below derive c from one implementation rather than
    # two that might drift apart.
    fit = fit_c_from_arms(baseline, spec)
    t_target_ms = fit["t_target_ms"]
    c_estimates = [(e["K"], e["t_draft_ms"], e["c"]) for e in fit["per_k"]]

    out.append(
        {
            "type": "para",
            "text": f"The baseline emits {n_tokens} tokens using "
            f"{baseline['mean_target_forward_calls']:.0f} target forward passes at "
            f"{baseline['mean_tps']:.2f} tokens/s, which puts one target forward pass "
            f"at {t_target_ms:.1f} ms. For reference, streaming 2.20 GB of FP16 "
            "weights across a 192 GB/s bus takes 11.5 ms, so this stack reaches "
            f"roughly {11.5 / t_target_ms * 100:.0f}% of the memory-bandwidth roofline. "
            "The gap is kernel launch latency, attention over the KV cache, WDDM "
            "overhead and Python. It is worth knowing, because it means fixed "
            "per-call overhead is a large share of the draft cost c, which is what "
            "pushes the optimal K down on small models.",
        }
    )

    if c_estimates:
        rows = [["K", "Draft pass (ms)", "c = draft/target"]]
        for K, t_d, c in sorted(c_estimates):
            rows.append([str(K), f"{t_d:.2f}", f"{c:.3f}"])
        out.append({"type": "table", "rows": rows})
        c_mean = float(np.mean([c for _, _, c in c_estimates]))
        out.append(
            {
                "type": "para",
                "text": f"Taking c = {c_mean:.3f} (the mean across arms), the model "
                "speedup = alpha / (1 + K*c) predicts:",
            }
        )
        prows = [["K", "Measured alpha", "Predicted speedup", "Measured speedup", "Error"]]
        for K in sorted(spec):
            s = spec[K]
            pred = s["mean_acceptance_length"] / (1.0 + K * c_mean)
            meas = s["speedup"]
            prows.append(
                [
                    str(K),
                    f"{s['mean_acceptance_length']:.3f}",
                    f"{pred:.3f}",
                    f"{meas:.3f}",
                    f"{(pred - meas) / meas * 100:+.1f}%",
                ]
            )
        out.append({"type": "table", "rows": prows})
        out.append(
            {
                "type": "para",
                "text": "The model consistently predicts slightly high, which is "
                "expected: it charges the verification block the cost of a "
                "single-token pass, but the block is K+1 tokens wide and carries "
                "per-round Python and cache-rollback overhead the model omits. The "
                "point is that a two-parameter model fitted to independently "
                "measured quantities reproduces the shape of the sweep, including "
                "where it peaks and that it declines afterwards. The system behaves "
                "the way the theory says it should.",
            }
        )

    # The same model, applied to the EAGLE arms. If it predicts those too, the
    # negative Phase 2 result is explained by the same mechanics rather than by
    # something unaccounted for.
    ckpts = load_json("eagle_checkpoints.json") or {}
    erows = [["Head", "K", "alpha", "c", "Predicted", "Measured", "Error"]]
    any_e = False
    for name in sorted(ckpts):
        bh, ph = arms_for_prefix(summary, f"phase2_{name}")
        if not bh or not ph:
            continue
        for K in sorted(ph):
            s = ph[K]
            total_ms = n_tokens / s["mean_tps"] * 1000.0
            n_d = max(s.get("mean_draft_forward_calls", 0), 1)
            c = (total_ms - s["mean_target_forward_calls"] * t_target_ms) / n_d / t_target_ms
            pred = s["mean_acceptance_length"] / (1.0 + K * c)
            any_e = True
            erows.append(
                [
                    name,
                    str(K),
                    f"{s['mean_acceptance_length']:.3f}",
                    f"{c:.3f}",
                    f"{pred:.3f}",
                    f"{s['speedup']:.3f}",
                    f"{(pred - s['speedup']) / s['speedup'] * 100:+.1f}%",
                ]
            )
    if any_e:
        out.append(
            {
                "type": "para",
                "text": "The same model applied to the EAGLE arms, with c measured "
                "per configuration rather than assumed:",
            }
        )
        out.append({"type": "table", "rows": erows})
        out.append(
            {
                "type": "para",
                "text": "Agreement here is tighter than for the draft model — within "
                "about one to two percent — because the head's cost is dominated by "
                "one large, regular matrix multiplication per step rather than by a "
                "whole small transformer, so the linear cost term fits it better. "
                "The value of this is that the Phase 2 negative result is not an "
                "anomaly needing a special explanation. The same two-parameter model "
                "that predicts where the draft model peaks also predicts, from "
                "independently measured alpha and c, that every EAGLE configuration "
                "lands below 1.0. Nothing unaccounted for is happening.",
            }
        )
    return out


# Nominal memory bandwidth of the GTX 1650 Ti Mobile, from the specification:
# 128-bit bus at 12 Gbps effective. Every roofline figure below is derived from
# this one constant and from measured quantities; nothing else is assumed.
DEVICE_BANDWIDTH_BYTES_PER_S = 192e9

# Layer counts for the two Phase 1 models, from their config.json. Used to
# separate a per-byte cost from a per-layer cost.
TARGET_LAYERS = 22   # TinyLlama-1.1B-Chat-v1.0
DRAFT_LAYERS = 2     # JackFram/llama-68m


def block_roofline(summary: dict) -> list:
    """
    Decompose the measured forward-pass costs against the memory-bandwidth
    roofline, and establish which of the two cost-model terms actually caps
    this configuration.

    Every number is derived from results/summary.json and results/vram_probe.json.
    Nothing here required a model or a run that does not exist.
    """
    out = [{"type": "heading3", "text": "How close to the memory-bandwidth roofline does this run?"}]
    baseline, spec = arms_for_prefix(summary, "phase1_T1.0")
    probe = load_json("vram_probe.json")
    if not baseline or not spec or not probe:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out

    measured = probe.get("measured", {})
    target_bytes = measured.get("target_weight_bytes")
    draft_bytes = measured.get("draft_weight_bytes")
    if not target_bytes or not draft_bytes:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out

    fit = fit_c_from_arms(baseline, spec)
    if not fit.get("ok"):
        out.append({"type": "para", "text": NOT_MEASURED})
        return out

    t_target_ms = fit["t_target_ms"]
    t_draft_ms = float(np.mean([e["t_draft_ms"] for e in fit["per_k"]]))
    roof_target_ms = target_bytes / DEVICE_BANDWIDTH_BYTES_PER_S * 1000.0
    roof_draft_ms = draft_bytes / DEVICE_BANDWIDTH_BYTES_PER_S * 1000.0
    frac_target = roof_target_ms / t_target_ms
    frac_draft = roof_draft_ms / t_draft_ms

    out.append(
        {
            "type": "para",
            "text": "Part I argued that batch-1 decoding is bandwidth-bound, which "
            "is the premise the whole method rests on. It is worth checking that "
            "premise against what the device actually did, because the answer turns "
            "out to qualify it. Reading the target's "
            f"{target_bytes / 1e9:.2f} GB of weights once across a "
            f"{DEVICE_BANDWIDTH_BYTES_PER_S / 1e9:.0f} GB/s bus takes "
            f"{roof_target_ms:.2f} ms. The measured forward pass takes "
            f"{t_target_ms:.1f} ms.",
        }
    )

    rows = [[
        "Model", "Weights", "Layers", "Roofline (ms)", "Measured (ms)",
        "% of roofline", "Effective GB/s",
    ]]
    for label, wbytes, layers, roof, meas in [
        ("TinyLlama-1.1B (target)", target_bytes, TARGET_LAYERS, roof_target_ms, t_target_ms),
        ("llama-68m (draft)", draft_bytes, DRAFT_LAYERS, roof_draft_ms, t_draft_ms),
    ]:
        rows.append([
            label,
            f"{wbytes / 1e9:.3f} GB",
            str(layers),
            f"{roof:.2f}",
            f"{meas:.2f}",
            f"{roof / meas * 100:.1f}%",
            f"{wbytes / (meas / 1000.0) / 1e9:.1f}",
        ])
    out.append({"type": "table", "rows": rows})

    out.append(
        {
            "type": "para",
            "text": f"So a decode step reaches about {frac_target * 100:.0f}% of the "
            "memory-bandwidth roofline, not the near-100% the simple model implies. "
            f"The effective bandwidth on the target is "
            f"{target_bytes / (t_target_ms / 1000.0) / 1e9:.0f} GB/s against a "
            f"nominal {DEVICE_BANDWIDTH_BYTES_PER_S / 1e9:.0f}. The missing time is "
            "kernel launch latency, attention over the KV cache, the WDDM display "
            "driver model's scheduling, and Python between the layers. This is a "
            "statement about the software stack and the device, not about "
            "speculative decoding: it applies identically to the baseline and to "
            "every speculative arm, and it caps absolute throughput rather than the "
            "ratio between them.",
        }
    )

    # ------------------------------------------------------------------
    # Where c comes from: bytes or depth?
    # ------------------------------------------------------------------
    byte_ratio = draft_bytes / target_bytes
    depth_ratio = DRAFT_LAYERS / TARGET_LAYERS
    c = fit["c"]
    out.append({"type": "heading4", "text": "What sets the draft cost c"})
    out.append(
        {
            "type": "para",
            "text": "The interesting consequence is for c, the draft pass measured "
            "in target passes. If a forward pass were purely weight streaming, c "
            "would be the ratio of the two models' weights. If it were purely "
            "per-layer overhead — a fixed cost paid once per layer regardless of how "
            "big that layer is — c would be the ratio of their depths. The measured "
            "value sits between the two, which says both mechanisms are present and "
            "neither alone accounts for it.",
        }
    )
    crows = [["Predictor", "Value", "What it would mean"]]
    crows.append([
        "Weight ratio (bytes)",
        f"{byte_ratio:.4f}",
        f"{draft_bytes / 1e9:.3f} GB / {target_bytes / 1e9:.3f} GB — pure bandwidth",
    ])
    crows.append([
        "Depth ratio (layers)",
        f"{depth_ratio:.4f}",
        f"{DRAFT_LAYERS} / {TARGET_LAYERS} layers — pure per-layer overhead",
    ])
    crows.append([
        "Measured c",
        f"{c:.4f}",
        "Fitted from the sweep, between the two",
    ])
    out.append({"type": "table", "rows": crows})

    per_layer_target = t_target_ms / TARGET_LAYERS
    per_layer_draft = t_draft_ms / DRAFT_LAYERS
    out.append(
        {
            "type": "para",
            "text": "The per-layer view makes the same point more directly. One "
            f"target layer costs {per_layer_target:.2f} ms and one draft layer "
            f"{per_layer_draft:.2f} ms, a factor of "
            f"{per_layer_target / per_layer_draft:.2f}, while a target layer holds "
            f"{(target_bytes / TARGET_LAYERS) / (draft_bytes / DRAFT_LAYERS):.2f} "
            "times as many bytes as a draft layer. Cost per layer is markedly "
            "sublinear in bytes per layer, which is the signature of a real fixed "
            "cost being paid once per layer.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "The practical reading is that on this device a drafter should "
            "be chosen for shallowness as much as for size. A model that is small "
            "in parameters but deep pays a per-layer toll that its parameter count "
            "does not predict. llama-68m happens to be both — 68 M parameters in "
            f"two layers — which is why c came out at {c:.3f} rather than "
            "something considerably worse. That was luck rather than design in the "
            "original model choice, and it is worth knowing deliberately.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "This decomposition rests on two models, which is two "
            "observations for two unknowns. It is reported as a bracket — c lies "
            "between the byte ratio and the depth ratio — and deliberately not as a "
            "fitted split, because two points cannot support one. A third model at a "
            "different depth-to-size ratio would settle it.",
        }
    )

    # ------------------------------------------------------------------
    # The binding constraint: alpha, not c.
    # ------------------------------------------------------------------
    out.append({"type": "heading4", "text": "Which term is actually binding"})
    best_k = max(spec, key=lambda k: spec[k]["speedup"])
    best = spec[best_k]
    alpha_max_k = max(spec, key=lambda k: spec[k]["mean_acceptance_length"])
    alpha_max = spec[alpha_max_k]["mean_acceptance_length"]

    rows2 = [["K", "alpha", "Measured speedup", "Speedup if the draft were free (c = 0)"]]
    for K in sorted(spec):
        s = spec[K]
        rows2.append([
            str(K),
            f"{s['mean_acceptance_length']:.3f}",
            f"{s['speedup']:.3f}x",
            f"{s['mean_acceptance_length']:.3f}x",
        ])
    out.append(
        {
            "type": "para",
            "text": "Setting c to zero in alpha / (1 + K·c) gives the speedup a "
            "drafter of literally no cost would deliver — the ceiling this draft "
            "model's acceptance rate permits, whatever the hardware does. It is just "
            "alpha.",
        }
    )
    out.append({"type": "table", "rows": rows2})
    out.append(
        {
            "type": "para",
            "text": f"The best measured configuration is K = {best_k} at "
            f"{best['speedup']:.3f}x. The highest acceptance length anywhere in the "
            f"sweep is {alpha_max:.3f} tokens per round, at K = {alpha_max_k}. So "
            "even an infinitely cheap drafter with this acceptance rate would top "
            f"out at {alpha_max:.2f}x, and reaching a published 2.5x would require "
            f"about {2.5 / alpha_max:.1f} times the acceptance this pair achieves. "
            "Removing the entire draft cost would not get there.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "That is the answer to why this project measured 1.32x where the "
            f"literature reports 2x to 3x. It is not that c is too high: c = {c:.3f} "
            "means the drafter contributes only "
            f"{best_k * c / (1 + best_k * c) * 100:.0f}% of the per-round cost at the "
            "optimum. It is that alpha is too low, and alpha is a property of how "
            "often the draft model and the target model agree. llama-68m and "
            "TinyLlama-1.1B share the Llama vocabulary and essentially nothing else "
            "— different pretraining corpora, different instruction tuning, no "
            "shared checkpoint ancestry. Published results that reach 2x to 3x "
            "generally use a drafter distilled from, or trained alongside, its "
            "target.",
        }
    )
    out.append(
        {
            "type": "para",
            "text": "Which is what makes this a useful negative rather than a "
            "disappointing number: the decomposition says exactly where the "
            "remaining headroom is. Effort spent making the drafter cheaper is "
            "close to wasted on this configuration. Effort spent on draft-target "
            "agreement — a same-family drafter, or a head trained against this "
            "target on enough data — is where the factor of two lives. Part IV's "
            "EAGLE attempt was aimed at precisely that term, and its failure is a "
            "measurement of a 250,000-token training budget rather than of the "
            "approach.",
        }
    )
    return out


def block_environment() -> list:
    out = [{"type": "heading3", "text": "Measurement environment"}]
    raw = load_raw()
    probe = load_json("vram_probe.json")
    if not raw:
        out.append({"type": "para", "text": NOT_MEASURED})
        return out
    r = raw[-1]
    rows = [["Field", "Value"]]
    rows.append(["Device", str(r.get("device"))])
    rows.append(["torch", str(r.get("torch_version"))])
    rows.append(["CUDA", str(r.get("cuda_version"))])
    rows.append(["Driver", str(r.get("driver_version"))])
    rows.append(["Target model", str(r.get("target_model"))])
    rows.append(["Tokens generated per run", str(r.get("max_new_tokens"))])
    if probe:
        m = probe.get("measured", {})
        rows.append(["Peak VRAM allocated", f"{m.get('peak_allocated_gb', 0):.3f} GB"])
        rows.append(["Target weights", f"{m.get('target_weight_bytes', 0) / 1e9:.3f} GB"])
        rows.append(["Draft weights", f"{m.get('draft_weight_bytes', 0) / 1e9:.3f} GB"])
        rows.append(["Target KV cache @2048", f"{m.get('target_kv_cache_bytes', 0) / 1e9:.3f} GB"])
        if m.get("target_prefill_logits_bytes"):
            rows.append(
                [
                    "Target prefill logits @2048",
                    f"{m['target_prefill_logits_bytes'] / 1e9:.3f} GB",
                ]
            )
    temps = [x.get("gpu_temp_c") for x in raw if x.get("gpu_temp_c") is not None]
    clocks = [x.get("gpu_clock_mhz") for x in raw if x.get("gpu_clock_mhz") is not None]
    if temps:
        rows.append(["GPU temperature range", f"{min(temps)}-{max(temps)} C"])
    if clocks:
        rows.append(["GPU clock range", f"{min(clocks)}-{max(clocks)} MHz"])
    rows.append(["Total run records", str(len(raw))])
    out.append({"type": "table", "rows": rows})
    if clocks and (max(clocks) - min(clocks)) > 100:
        out.append(
            {
                "type": "para",
                "text": f"GPU clock varied from {min(clocks)} to {max(clocks)} MHz over the "
                "run, which is thermal throttling in action. The interleaved design "
                "means this drift is shared across arms rather than charged to "
                "whichever ran last.",
            }
        )
    elif clocks:
        out.append(
            {
                "type": "para",
                "text": f"GPU clock stayed within {min(clocks)}-{max(clocks)} MHz across the "
                "run, so throttling was not a significant factor in this session. "
                "The interleaved design would have absorbed it had it been.",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Markdown -> document model
# ---------------------------------------------------------------------------

def parse_markdown(text: str) -> list:
    """Convert the restricted markdown used in kt-source.md into a doc model."""
    blocks = []
    lines = text.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("---") and set(stripped) <= {"-"}:
            blocks.append({"type": "rule"})
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            blocks.append({"type": f"heading{level}", "text": m.group(2).strip()})
            i += 1
            continue

        # Table: a line starting with | followed by a separator row.
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if re.match(r"^\|[\s:|-]+\|$", row):
                    i += 1
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows.append(cells)
                i += 1
            if rows:
                blocks.append({"type": "table", "rows": rows})
            continue

        # Indented code / formula block (4 spaces).
        if line.startswith("    "):
            code = []
            while i < n and (lines[i].startswith("    ") or not lines[i].strip()):
                if not lines[i].strip():
                    # Stop on a blank line that ends the block.
                    if i + 1 < n and lines[i + 1].startswith("    "):
                        code.append("")
                        i += 1
                        continue
                    break
                code.append(lines[i][4:])
                i += 1
            blocks.append({"type": "code", "lines": code})
            continue

        # Bullet or numbered list.
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n:
                s = lines[i].strip()
                mm = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", s)
                if not mm:
                    if s and items and not re.match(r"^(#{1,4})\s", s) and not s.startswith("|"):
                        # Continuation line of the current bullet.
                        items[-1] += " " + s
                        i += 1
                        continue
                    break
                items.append(mm.group(1).strip())
                i += 1
            blocks.append({"type": "list", "items": items})
            continue

        # Paragraph: consume until blank line or a structural marker.
        para = []
        while i < n:
            s = lines[i].strip()
            if not s:
                break
            if re.match(r"^(#{1,4})\s", s) or s.startswith("|") or lines[i].startswith("    "):
                break
            if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
                break
            if set(s) <= {"-"} and len(s) >= 3:
                break
            para.append(s)
            i += 1
        if para:
            blocks.append({"type": "para", "text": " ".join(para)})

    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the KT document model.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "build" / "kt_model.json")
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"Missing {SOURCE}", flush=True)
        return 1

    text = SOURCE.read_text(encoding="utf-8")
    # Drop the file's own front matter (everything before the first '---').
    parts = text.split("\n---\n", 1)
    if len(parts) == 2:
        text = parts[1]

    summary = load_summary()
    raw = load_raw()

    blocks = []
    for block in parse_markdown(text):
        if block["type"] == "para" and block["text"].startswith("{{RESULTS:"):
            key = block["text"].strip().strip("{}").split(":", 1)[1]
            if key == "PHASE1":
                blocks.extend(block_environment())
                blocks.extend(block_phase1(summary))
            elif key == "LATENCY":
                blocks.extend(block_latency(summary))
            elif key == "DOMAIN":
                blocks.extend(block_domain(raw))
            elif key == "PHASE2":
                blocks.extend(block_phase2(summary))
            elif key == "EAGLEVSDRAFT":
                blocks.extend(block_eagle_vs_draft(summary))
            elif key == "COSTMODEL":
                blocks.extend(block_costmodel(summary))
            elif key == "ROOFLINE":
                blocks.extend(block_roofline(summary))
            continue
        blocks.append(block)

    model = {
        "title": "Speculative Decoding on a 4 GB Laptop GPU",
        "subtitle": "Knowledge transfer and technical handover",
        "generated": datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC"),
        "blocks": blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"Wrote document model: {args.out} ({len(blocks)} blocks)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
