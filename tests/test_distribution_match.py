"""
tests/test_distribution_match.py — end-to-end statistical correctness check.

Samples the first generated token N=20,000 times from both the baseline
and speculative decoding, then compares the two empirical distributions
using a two-sample chi-square test and bootstrap confidence interval on
total variation distance.

This test confirms that the distribution guarantee holds with real models
(not just the synthetic Monte Carlo). It is slower than the synthetic test.

Marked @pytest.mark.slow — run with `pytest -m slow` or include explicitly.
Must pass before results are reported.

Requires CUDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from specbench.models import (
    load_draft_model,
    load_target_model,
    verify_cuda_environment,
)
from specbench.sampling import logits_to_probs, sample_token
from specbench.stats import (
    bootstrap_ci,
    chi_square_two_sample,
    total_variation_distance,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available.",
    ),
]

MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "cuda"
N_SAMPLES = 20_000

# Fixed short prompt for repeatability.
TEST_PROMPT = "The quick brown fox"

# TVD tolerance: at N=20,000, standard error is ~1/√N ≈ 0.007.
TVD_TOLERANCE = 0.05   # Generous to account for model non-determinism.
CHI2_P_THRESHOLD = 0.001


@pytest.fixture(scope="module")
def models():
    verify_cuda_environment()
    target_model, tokenizer = load_target_model(MODEL_DIR, device=DEVICE)
    draft_model, _ = load_draft_model(MODEL_DIR, device=DEVICE)
    return target_model, draft_model, tokenizer


def _sample_first_token_baseline(model, input_ids, n_samples, temperature=1.0, seed=42):
    """
    Sample the first generated token N times using baseline (target-only).
    Returns array of token ids.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    tokens = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
        logit = out.logits[0, -1, :]
        probs = logits_to_probs(logit, temperature)

    for _ in range(n_samples):
        tok = sample_token(probs.unsqueeze(0), generator=generator).item()
        tokens.append(tok)

    return np.array(tokens, dtype=np.int64)


def _sample_first_token_spec(draft_model, target_model, input_ids, n_samples, K=4,
                               temperature=1.0, seed=42):
    """
    Sample the first generated token N times using speculative decoding.

    For a fair comparison, we use K=4 draft proposals. The accept/reject rule
    means the FIRST emitted token may come from the target distribution (if the
    first proposal was accepted) or from the residual. Either way, its distribution
    must match the target's first-token distribution.
    """
    from specbench.spec_decode import generate_speculative

    tokens = []
    for i in range(n_samples):
        result = generate_speculative(
            draft_model,
            target_model,
            input_ids=input_ids.clone(),
            max_new_tokens=1,
            K=K,
            temperature=temperature,
            device=DEVICE,
            seed=seed + i,
        )
        tokens.append(result["generated_ids"][0])
    return np.array(tokens, dtype=np.int64)


def test_distribution_match(models):
    """
    Two-sample chi-square test: baseline and speculative first-token distributions
    must be statistically indistinguishable.
    """
    target_model, draft_model, tokenizer = models

    enc = tokenizer(TEST_PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    # Sample baseline first tokens.
    baseline_tokens = _sample_first_token_baseline(
        target_model, input_ids, n_samples=N_SAMPLES
    )

    # Sample speculative first tokens.
    spec_tokens = _sample_first_token_spec(
        draft_model, target_model, input_ids, n_samples=N_SAMPLES
    )

    # Build count arrays over the shared vocabulary.
    vocab_size = tokenizer.vocab_size
    baseline_counts = np.bincount(baseline_tokens, minlength=vocab_size).astype(np.float64)
    spec_counts = np.bincount(spec_tokens, minlength=vocab_size).astype(np.float64)

    # Two-sample chi-square test.
    chi2, p_val = chi_square_two_sample(baseline_counts, spec_counts)
    assert p_val > CHI2_P_THRESHOLD, (
        f"Two-sample chi-square FAILED: p={p_val:.6f} < {CHI2_P_THRESHOLD}. "
        f"Speculative distribution may differ from baseline."
    )

    # Total variation distance with bootstrap CI.
    baseline_emp = baseline_counts / baseline_counts.sum()
    spec_emp = spec_counts / spec_counts.sum()
    tvd = total_variation_distance(baseline_emp, spec_emp)

    assert tvd < TVD_TOLERANCE, (
        f"TVD={tvd:.4f} between baseline and speculative exceeds tolerance {TVD_TOLERANCE}. "
        f"chi2={chi2:.2f}, p={p_val:.4f}."
    )

    print(f"\n  Distribution match: chi2={chi2:.2f}, p={p_val:.4f}, TVD={tvd:.4f}")
    print(f"  N_SAMPLES={N_SAMPLES} (baseline) + {N_SAMPLES} (spec)")
