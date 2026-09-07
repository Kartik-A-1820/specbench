"""
scripts/run_vram_probe.py — Phase 0 VRAM verification.

Loads both models, runs a 2048-token forward pass with both resident,
and records predicted vs. measured VRAM allocations. Results are written
to results/vram_probe.json.

This script must be run and the results reviewed before Phase 1 begins.
If OOM occurs, reduce CONTEXT_LENGTH to 1024 per the Phase 0 instructions.

Usage:
    python scripts/run_vram_probe.py
    python scripts/run_vram_probe.py --context-length 1024
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Allow running from anywhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

from specbench.kv import make_empty_cache, prefill_cache
from specbench.models import (
    cuda_env_info,
    load_draft_model,
    load_target_model,
    reset_peak_memory,
    vram_report,
    verify_cuda_environment,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "vram_probe.json"


def run_vram_probe(context_length: int = 2048) -> dict:
    """
    Load both models and measure VRAM allocation at full context.

    Returns the probe results as a dict for JSON serialisation.
    """
    verify_cuda_environment()
    env = cuda_env_info()
    print(f"Device: {env['device']}")
    print(f"Torch: {env['torch_version']}, CUDA: {env['cuda_version']}")
    print(f"Driver: {env['driver_version']}")
    print(f"Total VRAM: {env['device_total_memory_bytes'] / 1e9:.2f} GB")

    DEVICE = "cuda"
    MODEL_DIR = Path(__file__).parent.parent / "models"

    torch.cuda.empty_cache()
    reset_peak_memory()

    # ----------------------------------------------------------------
    # Predicted values (from §3.1 of the build brief).
    # ----------------------------------------------------------------
    predicted = {
        "target_weights_gb": 2.20,
        "draft_weights_gb": 0.14,
        "target_kv_cache_gb": 0.05,    # at 2048 context
        "draft_kv_cache_gb": 0.01,
        "cuda_context_gb": 0.30,       # estimated; measured separately
        "transient_activations_gb": 0.10,
        "total_gb": 2.80,
        "context_length": context_length,
    }

    # ----------------------------------------------------------------
    # Measure: CUDA context alone (before any model loading).
    # ----------------------------------------------------------------
    _ = torch.zeros(1, device=DEVICE)  # Force CUDA context init.
    torch.cuda.synchronize()
    r_baseline = vram_report("cuda_context_only")
    measured_cuda_context_bytes = r_baseline["allocated_bytes"]

    # ----------------------------------------------------------------
    # Load target model.
    # ----------------------------------------------------------------
    print(f"\nLoading target model...")
    target_model, tokenizer = load_target_model(MODEL_DIR, device=DEVICE)
    r_target = vram_report("after_target_load")
    target_weight_bytes = r_target["allocated_bytes"] - measured_cuda_context_bytes
    print(f"  Target weights: {target_weight_bytes / 1e9:.3f} GB (predicted: {predicted['target_weights_gb']:.2f} GB)")

    # ----------------------------------------------------------------
    # Load draft model.
    # ----------------------------------------------------------------
    print("Loading draft model...")
    draft_model, _ = load_draft_model(MODEL_DIR, device=DEVICE)
    r_both = vram_report("after_both_models_load")
    draft_weight_bytes = r_both["allocated_bytes"] - r_target["allocated_bytes"]
    print(f"  Draft weights: {draft_weight_bytes / 1e9:.3f} GB (predicted: {predicted['draft_weights_gb']:.2f} GB)")

    # ----------------------------------------------------------------
    # Forward pass with full context on target.
    # ----------------------------------------------------------------
    print(f"Running {context_length}-token forward pass on target...")
    dummy_ids = torch.ones(1, context_length, dtype=torch.long, device=DEVICE)
    if tokenizer.bos_token_id is not None:
        dummy_ids[:, 0] = tokenizer.bos_token_id

    target_cache = make_empty_cache()
    _, target_cache = prefill_cache(target_model, dummy_ids, target_cache)
    r_target_kv = vram_report("after_target_forward")
    target_kv_bytes = r_target_kv["allocated_bytes"] - r_both["allocated_bytes"]
    print(f"  Target KV cache: {target_kv_bytes / 1e9:.3f} GB (predicted: {predicted['target_kv_cache_gb']:.2f} GB)")

    # ----------------------------------------------------------------
    # Forward pass with full context on draft.
    # ----------------------------------------------------------------
    print(f"Running {context_length}-token forward pass on draft...")
    draft_cache = make_empty_cache()
    _, draft_cache = prefill_cache(draft_model, dummy_ids, draft_cache)
    r_draft_kv = vram_report("after_draft_forward")
    draft_kv_bytes = r_draft_kv["allocated_bytes"] - r_target_kv["allocated_bytes"]
    print(f"  Draft KV cache: {draft_kv_bytes / 1e9:.3f} GB (predicted: {predicted['draft_kv_cache_gb']:.2f} GB)")

    peak_report = vram_report("peak")
    peak_bytes = peak_report["max_allocated_bytes"]
    free_bytes = peak_report["free_bytes"]
    total_bytes = peak_report["total_bytes"]

    print(f"\n--- Summary ---")
    print(f"  Peak VRAM allocated:  {peak_bytes / 1e9:.3f} GB")
    print(f"  Free VRAM remaining:  {free_bytes / 1e9:.3f} GB")
    print(f"  Total device VRAM:    {total_bytes / 1e9:.3f} GB")
    print(f"  Predicted total:      {predicted['total_gb']:.2f} GB")
    print(f"  Headroom vs 3.4 GB:   {(3.4e9 - peak_bytes) / 1e9:.3f} GB")

    result = {
        "env": env,
        "context_length": context_length,
        "predicted": predicted,
        "measured": {
            "cuda_context_bytes": measured_cuda_context_bytes,
            "target_weight_bytes": target_weight_bytes,
            "draft_weight_bytes": draft_weight_bytes,
            "target_kv_cache_bytes": target_kv_bytes,
            "draft_kv_cache_bytes": draft_kv_bytes,
            "peak_allocated_bytes": peak_bytes,
            "free_bytes": free_bytes,
            "total_device_bytes": total_bytes,
            "peak_allocated_gb": peak_bytes / 1e9,
            "headroom_vs_3_4gb": (3.4e9 - peak_bytes) / 1e9,
        },
        "within_budget": peak_bytes < 3.4e9,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 0 VRAM probe.")
    parser.add_argument("--context-length", type=int, default=2048)
    args = parser.parse_args()

    result = run_vram_probe(context_length=args.context_length)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {OUTPUT_FILE}")

    if not result["within_budget"]:
        print(
            "\nWARNING: Peak allocation exceeds 3.4 GB budget. "
            "Reduce context length with --context-length 1024 before proceeding."
        )
        sys.exit(1)
    else:
        print("\nPhase 0 PASSED: configuration fits within VRAM budget.")


if __name__ == "__main__":
    main()
