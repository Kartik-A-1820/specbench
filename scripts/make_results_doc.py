"""
scripts/make_results_doc.py — regenerate docs/results.md from measured data.

Every number in docs/results.md is produced by this script from
results/summary.json and results/raw_measurements.jsonl. Nothing is typed by
hand, which is the point: a results table that a human edits is a table that
can drift from the data or be filled in from memory. If a configuration has
not run, its row says NOT MEASURED rather than carrying a plausible number.

Usage:
    python scripts/make_results_doc.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results"
DOCS = PROJECT_ROOT / "docs"
SUMMARY = RESULTS / "summary.json"
RAW = RESULTS / "raw_measurements.jsonl"

# Which prompt indices belong to which domain, matching PROMPT_SET ordering in
# specbench/harness.py.
DOMAIN_RANGES = {"prose": range(0, 15), "code": range(15, 25), "instructional": range(25, 31)}


def load_summary() -> dict:
    if not SUMMARY.exists():
        return {}
    try:
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_raw() -> list:
    if not RAW.exists():
        return []
    out = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def fmt(value, spec: str = ".3f") -> str:
    if value is None:
        return "NOT MEASURED"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def speedup_table(summary: dict, prefix: str, title: str) -> list[str]:
    rows = {}
    baseline_tps = None
    for name, s in summary.items():
        if not name.startswith(prefix) or not isinstance(s, dict) or "mean_tps" not in s:
            continue
        if s.get("mode") == "baseline":
            baseline_tps = s["mean_tps"]
            continue
        rows[s.get("K", 0)] = s

    if not rows:
        return [f"### {title}", "", "NOT MEASURED — this configuration has not been run.", ""]

    lines = [
        f"### {title}",
        "",
        f"Baseline (target model only): **{fmt(baseline_tps, '.2f')} tok/s**",
        "",
        "| K | Spec (tok/s) | Speedup | 95% CI | Mean tokens/round | Target fwd/run | Draft fwd/run |",
        "|---|---|---|---|---|---|---|",
    ]
    for K in sorted(rows):
        s = rows[K]
        lines.append(
            f"| {K} | {fmt(s['mean_tps'], '.2f')} | {fmt(s['speedup'], '.3f')}x | "
            f"{fmt(s['speedup_ci_95_low'], '.3f')}–{fmt(s['speedup_ci_95_high'], '.3f')} | "
            f"{fmt(s['mean_acceptance_length'])} | "
            f"{fmt(s.get('mean_target_forward_calls'), '.1f')} | "
            f"{fmt(s.get('mean_draft_forward_calls'), '.1f')} |"
        )
    lines.append("")
    return lines


def latency_table(summary: dict, prefix: str) -> list[str]:
    base = None
    picks = {}
    for name, s in summary.items():
        if not name.startswith(prefix) or not isinstance(s, dict) or "mean_tps" not in s:
            continue
        if s.get("mode") == "baseline":
            base = s
        elif s.get("K") in (4,):
            picks[s["K"]] = s
    if base is None or not picks:
        return ["### Latency", "", "NOT MEASURED.", ""]

    spec = picks[sorted(picks)[0]]
    lines = [
        "### Latency (K=4)",
        "",
        "Two series are reported because they answer different questions. "
        "*Round latency* is the wall-clock duration of one decoding round — one "
        "token for the baseline, up to K+1 for the speculative path. *Token "
        "arrival gap* is the interval between consecutive tokens becoming "
        "available; tokens after the first in a speculative round have a gap of "
        "0 ms because they genuinely arrive together, out of the same "
        "verification pass. Neither is derived from a mean.",
        "",
        "| Metric | Baseline | Speculative (K=4) |",
        "|---|---|---|",
    ]
    for label, key in [
        ("Round latency p50 (ms)", "p50_ms"),
        ("Round latency p90 (ms)", "p90_ms"),
        ("Round latency p99 (ms)", "p99_ms"),
        ("Round latency mean (ms)", "mean_ms"),
        ("Round latency std (ms)", "std_ms"),
    ]:
        b = base.get("round_latency_percentiles", {}).get(key)
        s = spec.get("round_latency_percentiles", {}).get(key)
        lines.append(f"| {label} | {fmt(b, '.2f')} | {fmt(s, '.2f')} |")
    for label, key in [
        ("Token arrival gap p50 (ms)", "p50_ms"),
        ("Token arrival gap p90 (ms)", "p90_ms"),
        ("Token arrival gap p99 (ms)", "p99_ms"),
    ]:
        b = base.get("token_arrival_gap_percentiles", {}).get(key)
        s = spec.get("token_arrival_gap_percentiles", {}).get(key)
        lines.append(f"| {label} | {fmt(b, '.2f')} | {fmt(s, '.2f')} |")
    lines.append("")
    return lines


def domain_table(raw: list) -> list[str]:
    by_domain = defaultdict(list)
    for r in raw:
        if r.get("mode") != "spec" or r.get("K") != 4 or r.get("temperature") != 1.0:
            continue
        pid = r.get("prompt_id")
        for domain, rng in DOMAIN_RANGES.items():
            if pid in rng:
                by_domain[domain].append(r["mean_acceptance_length"])
    if not by_domain:
        return ["### Acceptance by domain (K=4)", "", "NOT MEASURED.", ""]

    lines = [
        "### Acceptance by domain (K=4, temperature 1.0)",
        "",
        "| Domain | Prompts | Mean tokens/round | Std |",
        "|---|---|---|---|",
    ]
    for domain in ["prose", "code", "instructional"]:
        vals = by_domain.get(domain, [])
        if not vals:
            lines.append(f"| {domain} | 0 | NOT MEASURED | — |")
            continue
        arr = np.array(vals)
        lines.append(
            f"| {domain} | {len(arr)} | {arr.mean():.3f} | {arr.std():.3f} |"
        )
    lines.append("")
    return lines


def environment_block(raw: list) -> list[str]:
    if not raw:
        return ["No run records — the benchmark has not been executed.", ""]
    r = raw[-1]
    probe = {}
    probe_path = RESULTS / "vram_probe.json"
    if probe_path.exists():
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            probe = {}
    peak = probe.get("measured", {}).get("peak_allocated_gb")
    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| Device | {r.get('device')} |",
        f"| torch | {r.get('torch_version')} |",
        f"| CUDA | {r.get('cuda_version')} |",
        f"| Driver | {r.get('driver_version')} |",
        f"| Target model | {r.get('target_model')} |",
        f"| Max new tokens | {r.get('max_new_tokens')} |",
        f"| Peak VRAM (probe) | {fmt(peak, '.3f')} GB |",
        f"| Run records | {len(raw)} |",
        "",
    ]
    return lines


def main() -> int:
    summary = load_summary()
    raw = load_raw()

    if not summary and not raw:
        print(
            "No results on disk. docs/results.md left untouched — it will not be "
            "filled with placeholder numbers.",
            flush=True,
        )
        return 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Results",
        "",
        f"Generated by `scripts/make_results_doc.py` on {now}. Do not edit by hand:",
        "re-running the script overwrites this file. Every number below is computed",
        "from `results/summary.json` and `results/raw_measurements.jsonl`. Any",
        "configuration that has not been run shows NOT MEASURED.",
        "",
        "## Environment",
        "",
    ]
    lines += environment_block(raw)

    lines += ["## Phase 1 — separate draft model (llama-68m)", ""]
    lines += speedup_table(summary, "phase1_T1.0", "Sampling, temperature 1.0")
    lines += speedup_table(summary, "phase1_T0.0", "Greedy, temperature 0.0")
    lines += latency_table(summary, "phase1_T1.0")
    lines += domain_table(raw)

    lines += ["## Phase 2 — EAGLE draft head", ""]
    ckpt_path = RESULTS / "eagle_checkpoints.json"
    if ckpt_path.exists():
        ckpts = json.loads(ckpt_path.read_text(encoding="utf-8"))
        lines += [
            "### Trained heads",
            "",
            "| Head | Layers | Intermediate | Params | Final CE loss | Peak VRAM (GB) |",
            "|---|---|---|---|---|---|",
        ]
        for name in sorted(ckpts):
            c = ckpts[name]
            lines.append(
                f"| {name} | {c.get('num_layers')} | {c.get('intermediate_size')} | "
                f"{c.get('parameters', 0) / 1e6:.1f} M | "
                f"{fmt(c.get('final_ce_loss'), '.4f')} | "
                f"{fmt((c.get('peak_vram_bytes') or 0) / 1e9, '.3f')} |"
            )
        lines.append("")
        for name in sorted(ckpts):
            lines += speedup_table(summary, f"phase2_{name}", f"EAGLE head {name}")
    else:
        lines += ["NOT MEASURED — no EAGLE head has been trained.", ""]

    lines += [
        "## Reproducing any number",
        "",
        "```python",
        "import json, numpy as np",
        "from specbench.stats import bootstrap_speedup_ci",
        "",
        'records = [json.loads(l) for l in open("results/raw_measurements.jsonl")]',
        'base = [r for r in records if r["arm"].endswith("baseline")]',
        'spec = [r for r in records if r["mode"] == "spec" and r["K"] == 4]',
        'b = np.array([r["tokens_per_s"] for r in base])',
        's = np.array([r["tokens_per_s"] for r in spec])',
        "print(bootstrap_speedup_ci(b, s))",
        "```",
        "",
    ]

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DOCS / 'results.md'} ({len(lines)} lines).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
