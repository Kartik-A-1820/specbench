"""
tests/test_sampler_synthetic.py — the core correctness proof.

This is the most important test in the repository. It isolates the accept/reject
rule from the models entirely and proves the sampler is distribution-preserving
by Monte Carlo simulation over synthetic probability distributions.

No CUDA, no model loading — runs on CPU in seconds and belongs in CI.

Test cases include:
  - Standard case: arbitrary p and q over 8 tokens.
  - q == p: acceptance rate must be 1.0 (every proposal accepted).
  - q uniform: distribution guarantee holds regardless of draft quality.
  - Adversarial: near-zero q mass where p has mass; disjoint support.
  - Heavily peaked p.

For each case: N >= 1,000,000 trials, chi-square goodness-of-fit p > 0.001,
total variation distance within Monte-Carlo error (~1/√N), mean acceptance
length within tolerance of closed-form expectation.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
import torch

from specbench.spec_decode import compute_acceptance, sample_residual
from specbench.stats import chi_square_goodness_of_fit, total_variation_distance

# Number of Monte Carlo trials. 1e6 gives TVD error ~1e-3.
N_TRIALS = 1_000_000

# Chi-square p-value threshold.
CHI2_P_THRESHOLD = 0.001

# TVD tolerance: ~3/√N at N=1e6 gives 0.003 (generous for adversarial cases).
TVD_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# Reference implementation of one speculative round (CPU, no models)
# ---------------------------------------------------------------------------

def run_synthetic_round(
    p: torch.Tensor,
    q: torch.Tensor,
    K: int,
    n_trials: int,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Run n_trials independent speculative decoding rounds using synthetic p and q.

    Each round: draft K tokens from q, accept/reject against p, return emitted tokens
    and acceptance counts.

    Returns:
        (emitted_tokens_flat, accepted_counts_per_trial)
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    vocab = p.shape[0]
    emitted_all = []
    accepted_counts = []

    for _ in range(n_trials):
        emitted = []
        accepted = 0

        # Draft K tokens from q.
        proposed = torch.multinomial(q, num_samples=K, replacement=True, generator=rng)

        for i in range(K):
            x_i = proposed[i].item()
            accepted_flag = compute_acceptance(p, q, x_i, generator=None)

            if accepted_flag:
                emitted.append(x_i)
                accepted += 1
            else:
                # Residual sampling.
                x_prime, _ = sample_residual(p, q, generator=None)
                emitted.append(x_prime)
                break

        if accepted == K:
            # Bonus token from p.
            x_bonus = torch.multinomial(p, num_samples=1, generator=rng).item()
            emitted.append(x_bonus)

        emitted_all.extend(emitted)
        accepted_counts.append(accepted)

    return emitted_all, accepted_counts


def check_distribution(
    emitted_tokens: list,
    p: torch.Tensor,
    label: str,
) -> None:
    """
    Assert the empirical distribution of emitted_tokens matches p.

    Checks:
      1. Chi-square goodness-of-fit p-value > CHI2_P_THRESHOLD.
      2. Total variation distance < TVD_TOLERANCE.
    """
    vocab = p.shape[0]
    p_np = p.numpy().astype(np.float64)
    p_np = p_np / p_np.sum()  # Renormalise for safety.

    counts = np.zeros(vocab, dtype=np.float64)
    for tok in emitted_tokens:
        counts[tok] += 1

    empirical = counts / counts.sum()
    tvd = total_variation_distance(empirical, p_np)

    chi2, p_val = chi_square_goodness_of_fit(counts, p_np)

    assert p_val > CHI2_P_THRESHOLD, (
        f"[{label}] Chi-square test FAILED: p={p_val:.6f} < {CHI2_P_THRESHOLD}. "
        f"Empirical dist may differ from target p."
    )
    assert tvd < TVD_TOLERANCE, (
        f"[{label}] TVD={tvd:.4f} exceeds tolerance {TVD_TOLERANCE}."
    )


# ---------------------------------------------------------------------------
# Closed-form expected acceptance length
# ---------------------------------------------------------------------------

def expected_acceptance_length(p: torch.Tensor, q: torch.Tensor, K: int) -> float:
    """
    Compute the closed-form expected acceptance length.

    Expected alpha = sum_{i=1}^{K} prod_{j=1}^{i} P(accept_j) + K * prod_{j=1}^{K} P(accept_j)

    where P(accept_j) = sum_x min(p(x), q(x)) = 1 - TVD(p, q) ... actually:
    P(accept at position j) = sum_x min(p(x), q(x)) = 1 - 0.5*sum_x|p-q|... no.

    Correct: The probability of accepting a single token proposed by q is:
        sum_x q(x) * min(1, p(x)/q(x)) = sum_x min(q(x), p(x))

    Let beta = sum_x min(p(x), q(x)).
    Expected number of accepted tokens (not counting bonus) = sum_{i=1}^{K} beta^i / beta^{i-1}...

    Actually: the positions are independent draws.
    P(accept position i | all i-1 accepted) depends on the conditioned proposal —
    but for i.i.d. draws from q, P(accept any proposal) = beta = sum min(p,q).

    Expected tokens emitted per round = sum_{i=0}^{K-1} beta^i + K * beta^K + 1 - K*beta^K... hmm.

    Standard result (Leviathan et al. eq. 3):
        E[emitted] = (1 - beta^{K+1}) / (1 - beta)  if beta < 1
                   = K + 1                            if beta = 1
    """
    p_np = p.numpy().astype(np.float64)
    q_np = q.numpy().astype(np.float64)
    beta = float(np.sum(np.minimum(p_np, q_np)))
    if abs(beta - 1.0) < 1e-9:
        return float(K + 1)
    return (1.0 - beta ** (K + 1)) / (1.0 - beta)


# ---------------------------------------------------------------------------
# Test distributions
# ---------------------------------------------------------------------------

def make_distributions():
    """
    Return a list of (p, q, label) tuples covering standard and adversarial cases.
    """
    vocab = 8

    cases = []

    # 1. Uniform q, arbitrary p.
    p1 = torch.tensor([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05], dtype=torch.float32)
    q1 = torch.ones(vocab) / vocab
    cases.append((p1, q1, "uniform_q"))

    # 2. q == p: acceptance rate must be 1.0.
    p2 = torch.tensor([0.4, 0.3, 0.15, 0.07, 0.03, 0.02, 0.02, 0.01], dtype=torch.float32)
    q2 = p2.clone()
    cases.append((p2, q2, "q_equals_p"))

    # 3. Near-disjoint support: q puts mass mainly where p has little.
    p3 = torch.tensor([0.5, 0.4, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005], dtype=torch.float32)
    q3 = torch.tensor([0.01, 0.01, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01], dtype=torch.float32)
    q3 = q3 / q3.sum()
    cases.append((p3, q3, "near_disjoint_support"))

    # 4. Heavily peaked p, uniform q.
    p4 = torch.zeros(vocab)
    p4[0] = 0.9
    p4[1] = 0.09
    p4[2] = 0.01
    q4 = torch.ones(vocab) / vocab
    cases.append((p4, q4, "peaked_p_uniform_q"))

    # 5. q near-zero where p has mass at tokens 0-1.
    p5 = torch.tensor([0.45, 0.45, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01], dtype=torch.float32)
    q5 = torch.tensor([0.001, 0.001, 0.3, 0.3, 0.2, 0.1, 0.05, 0.048], dtype=torch.float32)
    q5 = q5 / q5.sum()
    cases.append((p5, q5, "q_zero_where_p_has_mass"))

    # 6. Both distributions smooth.
    p6 = torch.softmax(torch.arange(vocab, dtype=torch.float32) * 0.5, dim=0)
    q6 = torch.softmax(torch.arange(vocab, dtype=torch.float32) * 0.3, dim=0)
    cases.append((p6, q6, "smooth_both"))

    return cases


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,q,label", make_distributions())
def test_distribution_preserving(p, q, label):
    """
    Core correctness test: emitted token distribution must match p.

    Runs N_TRIALS rounds at K=4, checks chi-square and TVD.
    """
    K = 4
    emitted, _ = run_synthetic_round(p, q, K=K, n_trials=N_TRIALS)
    check_distribution(emitted, p, label)


def test_q_equals_p_acceptance_rate():
    """
    When q == p, every proposal must be accepted (acceptance rate = 1.0).
    """
    vocab = 8
    p = torch.tensor([0.4, 0.3, 0.15, 0.07, 0.03, 0.02, 0.02, 0.01], dtype=torch.float32)
    q = p.clone()
    K = 4
    _, accepted_counts = run_synthetic_round(p, q, K=K, n_trials=10_000, seed=0)
    mean_accepted = sum(accepted_counts) / len(accepted_counts)
    # q == p means beta = 1; every proposal should be accepted (all K tokens each round).
    assert mean_accepted == pytest.approx(K, abs=0.01), (
        f"q==p but mean accepted per round = {mean_accepted:.3f}, expected {K}."
    )


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_mean_acceptance_length(K):
    """
    Mean tokens emitted per round must match closed-form expectation.
    """
    p = torch.tensor([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05], dtype=torch.float32)
    q = torch.ones(8) / 8

    emitted, _ = run_synthetic_round(p, q, K=K, n_trials=N_TRIALS, seed=K)
    expected = expected_acceptance_length(p, q, K)

    # Estimate emitted per round: total emitted / n_trials.
    emitted_per_round = len(emitted) / N_TRIALS
    tolerance = 3 * math.sqrt(K / N_TRIALS)  # generous for outliers
    assert abs(emitted_per_round - expected) < max(tolerance, 0.05), (
        f"K={K}: emitted/round={emitted_per_round:.4f}, expected={expected:.4f}, "
        f"tol={tolerance:.4f}"
    )


def test_adversarial_near_disjoint_distribution():
    """
    Even with near-disjoint support, output distribution must still match p.
    """
    vocab = 8
    p = torch.tensor([0.5, 0.4, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005], dtype=torch.float32)
    q = torch.tensor([0.01, 0.01, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01], dtype=torch.float32)
    q = q / q.sum()

    emitted, _ = run_synthetic_round(p, q, K=4, n_trials=N_TRIALS, seed=99)
    check_distribution(emitted, p, "adversarial_near_disjoint")


@pytest.mark.parametrize("K", [1, 4])
def test_uniform_q_distribution(K):
    """
    With uniform q, the guarantee holds; speed degrades but correctness does not.
    """
    vocab = 8
    p = torch.tensor([0.5, 0.3, 0.1, 0.04, 0.02, 0.02, 0.01, 0.01], dtype=torch.float32)
    q = torch.ones(vocab) / vocab

    emitted, _ = run_synthetic_round(p, q, K=K, n_trials=N_TRIALS, seed=7)
    check_distribution(emitted, p, f"uniform_q_K{K}")
