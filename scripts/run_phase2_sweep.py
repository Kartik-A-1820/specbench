"""
scripts/run_phase2_sweep.py — Phase 2 step 3: benchmark the trained EAGLE heads.

For each trained head, runs the interleaved multi-arm benchmark with the
baseline plus one arm per speculation depth. Results go to the same
raw_measurements.jsonl schema as Phase 1 and are merged into summary.json, so
Phase 1 and Phase 2 points land on one Pareto plot.

Resumability: one head at a time, each writing
results/phase2_<head>_summary.json. A head whose summary file already exists is
skipped. An interrupted sweep therefore resumes at the next head rather than
repeating hours of work.

Usage:
    python scripts/run_phase2_sweep.py
    python scripts/run_phase2_sweep.py --k-values 2 3 4 6 --heads eagle_L1_I5632
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from specbench.baseline import generate_baseline
from specbench.eagle.head import EagleDraftHead
from specbench.eagle.spec_decode_eagle import generate_eagle
from specbench.harness import Arm, merge_summary, run_multiarm_benchmark
from specbench.models import load_target_model, verify_cuda_environment

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

TARGET_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    parser = argparse.ArgumentParser(description="Phase 2 EAGLE benchmark sweep.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--heads", type=str, nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--done-marker", type=str, default=None)
    args = parser.parse_args()

    manifest_path = RESULTS_DIR / "eagle_checkpoints.json"
    if not manifest_path.exists():
        print(
            f"No checkpoint manifest at {manifest_path}. Run "
            "scripts/run_phase2_train.py first.",
            flush=True,
        )
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    head_names = args.heads or sorted(manifest)
    head_names = [h for h in head_names if h in manifest]
    if not head_names:
        print("No trained heads to benchmark.", flush=True)
        sys.exit(1)

    verify_cuda_environment()

    print("Loading target model...", flush=True)
    target_model, tokenizer = load_target_model(MODEL_DIR, device="cuda")
    target_config = target_model.config
    token_embedding = target_model.model.embed_tokens
    lm_head_weight = target_model.lm_head.weight
    print("Target model loaded.", flush=True)

    all_summaries = {}

    for head_name in head_names:
        out_path = RESULTS_DIR / f"phase2_{head_name}_summary.json"
        if out_path.exists():
            print(f"{head_name}: summary already exists, skipping.", flush=True)
            all_summaries.update(json.loads(out_path.read_text(encoding="utf-8")))
            continue

        info = manifest[head_name]
        print(f"\n{'=' * 60}\nBenchmarking {head_name}\n{'=' * 60}", flush=True)

        head = EagleDraftHead(
            config=target_config,
            num_draft_layers=info["num_layers"],
            intermediate_size=info["intermediate_size"],
        )
        head.load_state_dict(torch.load(info["checkpoint"], map_location="cpu"))
        head = head.to("cuda").float().eval()

        def make_arm(K: int, head=head) -> Arm:
            def fn(**kw):
                return generate_eagle(
                    target_model, head, token_embedding, lm_head_weight, K=K, **kw
                )

            return Arm(
                name=f"phase2_{head_name}_K{K}",
                mode="spec_eagle",
                fn=fn,
                K=K,
                draft_model_name=head_name,
            )

        arms = [
            Arm(
                name=f"phase2_{head_name}_baseline",
                mode="baseline",
                fn=lambda **kw: generate_baseline(target_model, **kw),
                K=0,
                draft_model_name="",
                is_baseline=True,
            )
        ] + [make_arm(K) for K in args.k_values]

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
            print(f"FAILED for {head_name}: {summaries['error']}", flush=True)
            sys.exit(1)

        for name, s in summaries.items():
            s["head_parameters"] = info.get("parameters")
            s["head_num_layers"] = info.get("num_layers")
            s["head_intermediate_size"] = info.get("intermediate_size")
            merge_summary(name, s)
            print(
                f"  {name:40s} {s['mean_tps']:7.2f} tok/s  "
                f"speedup {s['speedup_str']:26s} alpha {s['mean_acceptance_length']:.3f}",
                flush=True,
            )

        out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        all_summaries.update(summaries)

        del head
        torch.cuda.empty_cache()

    combined = RESULTS_DIR / "phase2_sweep_summary.json"
    combined.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nCombined sweep summary written to {combined}", flush=True)

    if args.done_marker:
        Path(args.done_marker).write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
