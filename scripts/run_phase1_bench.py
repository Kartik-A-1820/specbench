"""
scripts/run_phase1_bench.py — Phase 1 interleaved benchmark.

Runs the target-only baseline and speculative decoding at several speculation
depths over the full prompt set, using the interleaved multi-arm design in
specbench/harness.py: for each prompt every arm runs once, back to back, with
the arm order rotated from prompt to prompt.

All arms share one baseline. Running a separate baseline per K would be six
identical baselines measured at six different points in a thermal ramp, which
is both slower and noisier than measuring it once per prompt.

Per-run records are appended to results/raw_measurements.jsonl. Aggregates are
merged into results/summary.json under a tag per configuration, so an
interrupted run keeps everything it already finished.

Usage:
    python scripts/run_phase1_bench.py
    python scripts/run_phase1_bench.py --k-values 1 2 4 8 --temperature 1.0
    python scripts/run_phase1_bench.py --temperature 0.0 --tag phase1_greedy
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from specbench.baseline import generate_baseline
from specbench.harness import Arm, merge_summary, run_multiarm_benchmark
from specbench.models import load_draft_model, load_target_model, verify_cuda_environment
from specbench.spec_decode import generate_speculative

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DRAFT_MODEL_NAME = "JackFram/llama-68m"


def main():
    parser = argparse.ArgumentParser(description="Phase 1 speculative decoding benchmark.")
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8],
        help="Speculation depths to measure as separate arms.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Prefix for summary.json keys. Defaults to phase1_T<temperature>.",
    )
    parser.add_argument(
        "--done-marker", type=str, default=None,
        help="File to touch on success, used by run_all.bat for resumability.",
    )
    args = parser.parse_args()

    tag_prefix = args.tag or f"phase1_T{args.temperature}"

    verify_cuda_environment()

    print("Loading models...", flush=True)
    target_model, tokenizer = load_target_model(MODEL_DIR, device="cuda")
    draft_model, _ = load_draft_model(MODEL_DIR, device="cuda")
    print("Models loaded.", flush=True)

    def make_spec_arm(K: int) -> Arm:
        def fn(**kw):
            return generate_speculative(draft_model, target_model, K=K, **kw)

        return Arm(
            name=f"{tag_prefix}_spec_K{K}",
            mode="spec",
            fn=fn,
            K=K,
            draft_model_name=DRAFT_MODEL_NAME,
        )

    arms = [
        Arm(
            name=f"{tag_prefix}_baseline",
            mode="baseline",
            fn=lambda **kw: generate_baseline(target_model, **kw),
            K=0,
            draft_model_name="",
            is_baseline=True,
        )
    ] + [make_spec_arm(K) for K in args.k_values]

    print(
        f"Arms: {len(arms)} ({', '.join(a.name for a in arms)})\n"
        f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    summaries = run_multiarm_benchmark(
        arms=arms,
        tokenizer=tokenizer,
        target_model_name=TARGET_MODEL_NAME,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed_base=args.seed,
        device="cuda",
        n_warmup=args.n_warmup,
    )

    if "error" in summaries:
        print(f"FAILED: {summaries['error']}", flush=True)
        sys.exit(1)

    print("\n--- Phase 1 results ---", flush=True)
    for name, s in summaries.items():
        merge_summary(name, s)
        lat = s.get("round_latency_percentiles", {})
        print(
            f"  {name:36s} {s['mean_tps']:7.2f} tok/s  "
            f"speedup {s['speedup_str']:26s} "
            f"alpha {s['mean_acceptance_length']:.3f}  "
            f"round p50/p90/p99 "
            f"{lat.get('p50_ms', 0):.1f}/{lat.get('p90_ms', 0):.1f}/{lat.get('p99_ms', 0):.1f} ms",
            flush=True,
        )

    out = RESULTS_DIR / f"{tag_prefix}_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWritten to {out}", flush=True)

    if args.done_marker:
        Path(args.done_marker).parent.mkdir(parents=True, exist_ok=True)
        Path(args.done_marker).write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
