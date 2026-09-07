"""
tests/test_vram_budget.py — VRAM budget regression test.

Assert that peak allocated memory with both models resident and a full-length
KV cache stays below the hard threshold of 3.4 GB.

This converts the VRAM budget from a claim into a regression test. Any change
to the loading configuration that pushes past the budget will be caught here.

Requires CUDA.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specbench.models import (
    load_draft_model,
    load_target_model,
    reset_peak_memory,
    vram_report,
    verify_cuda_environment,
)
from specbench.kv import make_empty_cache, prefill_cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available.",
)

MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "cuda"

# Hard VRAM limit: 3.4 GB (confirmed usable on GTX 1650 Ti after WDDM overhead).
VRAM_LIMIT_BYTES = 3.4 * 1024 ** 3

# Test with 2048 tokens as per the specification.
CONTEXT_LENGTH = 2048


@pytest.fixture(scope="module")
def both_models():
    verify_cuda_environment()
    target_model, tokenizer = load_target_model(MODEL_DIR, device=DEVICE)
    draft_model, _ = load_draft_model(MODEL_DIR, device=DEVICE)
    return target_model, draft_model, tokenizer


def test_peak_vram_both_models_resident(both_models):
    """
    With both models loaded and a full 2048-token KV cache active,
    peak allocated memory must stay below 3.4 GB.
    """
    target_model, draft_model, tokenizer = both_models

    # Reset peak tracker.
    torch.cuda.empty_cache()
    reset_peak_memory()

    # Allocate a 2048-token input (or as close as the tokenizer supports).
    # Use a repeated token sequence.
    dummy_ids = torch.ones(1, CONTEXT_LENGTH, dtype=torch.long, device=DEVICE)
    dummy_ids[:, 0] = tokenizer.bos_token_id or 1

    # Run a forward pass on the target with the full context.
    target_cache = make_empty_cache()
    _, target_cache = prefill_cache(target_model, dummy_ids, target_cache)

    # Run a forward pass on the draft with the full context.
    draft_cache = make_empty_cache()
    _, draft_cache = prefill_cache(draft_model, dummy_ids, draft_cache)

    report = vram_report("both_models_full_context")
    peak_bytes = report["max_allocated_bytes"]

    assert peak_bytes < VRAM_LIMIT_BYTES, (
        f"Peak VRAM {peak_bytes / 1e9:.2f} GB exceeds hard limit "
        f"{VRAM_LIMIT_BYTES / 1e9:.2f} GB. "
        f"The configuration does not fit on GTX 1650 Ti."
    )

    print(f"\n  Peak VRAM: {peak_bytes / 1e9:.3f} GB (limit: {VRAM_LIMIT_BYTES / 1e9:.1f} GB)")
    print(f"  Headroom: {(VRAM_LIMIT_BYTES - peak_bytes) / 1e9:.3f} GB")


def test_target_model_weight_footprint(both_models):
    """
    Target model weights should be approximately 2.2 GB in FP16 (1.1B params × 2 bytes).
    """
    target_model, _, _ = both_models

    param_bytes = sum(
        p.numel() * p.element_size() for p in target_model.parameters()
    )
    # Expected: 1.1B × 2 bytes = 2.2 GB. Allow ±15% for architectural overhead.
    expected_bytes = 2.2e9
    tolerance = 0.15

    assert abs(param_bytes - expected_bytes) / expected_bytes < tolerance, (
        f"Target model parameter footprint {param_bytes / 1e9:.2f} GB "
        f"deviates >15% from expected {expected_bytes / 1e9:.2f} GB."
    )


def test_draft_model_weight_footprint(both_models):
    """
    Draft model weights should be approximately 0.14 GB in FP16 (68M params × 2 bytes).
    """
    _, draft_model, _ = both_models

    param_bytes = sum(
        p.numel() * p.element_size() for p in draft_model.parameters()
    )
    expected_bytes = 0.14e9
    tolerance = 0.20  # Larger tolerance for small model.

    assert abs(param_bytes - expected_bytes) / expected_bytes < tolerance, (
        f"Draft model parameter footprint {param_bytes / 1e9:.3f} GB "
        f"deviates >20% from expected {expected_bytes / 1e9:.3f} GB."
    )
