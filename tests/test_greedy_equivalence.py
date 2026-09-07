"""
tests/test_greedy_equivalence.py — bit-identical output vs. baseline at temperature=0.

With temperature=0 (greedy decoding), speculative decoding must produce
token-for-token identical output to the autoregressive baseline on the same
prompt. This is a strong, fully deterministic check that immediately catches:
  - Off-by-one indexing in the KV cache or logits.
  - Cache rollback errors.
  - Logits-position mistakes.

Runs over 10 prompts, 128 new tokens each, at K in {1, 2, 4, 8}.
Any mismatch is a hard failure.

Requires CUDA (both models loaded to GPU).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch

from specbench.baseline import generate_baseline
from specbench.models import (
    load_draft_model,
    load_target_model,
    verify_cuda_environment,
)
from specbench.spec_decode import generate_speculative

# Skip all tests if CUDA unavailable (e.g. CI without GPU).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available — GPU required for greedy equivalence tests.",
)

MODEL_DIR = Path(__file__).parent.parent / "models"
MAX_NEW_TOKENS = 128
DEVICE = "cuda"

TEST_PROMPTS: List[str] = [
    "The capital of France is",
    "Once upon a time there was a",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "The speed of light in a vacuum is approximately",
    "In machine learning, overfitting occurs when",
    "The water cycle consists of several stages:",
    "import numpy as np\nnp.array([1, 2, 3])",
    "The three branches of government in the United States are",
    "Photosynthesis is the process by which",
    "The Pythagorean theorem states that in a right triangle,",
]


@pytest.fixture(scope="module")
def models():
    """Load target and draft models once for all tests in this module."""
    verify_cuda_environment()
    target_model, target_tokenizer = load_target_model(MODEL_DIR, device=DEVICE)
    draft_model, _ = load_draft_model(MODEL_DIR, device=DEVICE)
    return target_model, draft_model, target_tokenizer


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_greedy_equivalence(models, K):
    """
    Speculative decoding at temperature=0 must match baseline token-for-token.
    """
    target_model, draft_model, tokenizer = models
    eos_id = tokenizer.eos_token_id

    mismatches = []
    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)

        # Baseline: target-only greedy.
        baseline_result = generate_baseline(
            target_model,
            input_ids=input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            device=DEVICE,
            eos_token_id=eos_id,
        )

        # Speculative: same prompt, same greedy setting.
        spec_result = generate_speculative(
            draft_model,
            target_model,
            input_ids=input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            K=K,
            temperature=0.0,
            device=DEVICE,
            eos_token_id=eos_id,
        )

        baseline_ids = baseline_result["generated_ids"]
        spec_ids = spec_result["generated_ids"]

        # Trim to the shorter output (EOS may cause different lengths).
        min_len = min(len(baseline_ids), len(spec_ids))
        if baseline_ids[:min_len] != spec_ids[:min_len]:
            mismatches.append({
                "prompt_idx": prompt_idx,
                "K": K,
                "baseline": baseline_ids[:min_len],
                "spec": spec_ids[:min_len],
                "first_diff": next(
                    i for i, (a, b) in enumerate(zip(baseline_ids, spec_ids)) if a != b
                ),
            })

    assert not mismatches, (
        f"Greedy equivalence FAILED for K={K}. "
        f"{len(mismatches)}/{len(TEST_PROMPTS)} prompts mismatched. "
        f"First mismatch: prompt {mismatches[0]['prompt_idx']}, "
        f"position {mismatches[0]['first_diff']}."
    )
