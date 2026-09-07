"""
specbench/stats.py — bootstrap confidence intervals, chi-square tests,
and total variation distance.

All statistical machinery used by the benchmark harness and correctness tests
lives here. No model code. Works on CPU.

References:
  - Efron & Tibshirani (1993), "An Introduction to the Bootstrap".
  - The chi-square goodness-of-fit and two-sample tests use scipy.stats.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats as ss


# ---------------------------------------------------------------------------
# Total variation distance
# ---------------------------------------------------------------------------

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the total variation distance between two probability distributions.

    TVD(p, q) = 0.5 * sum |p_i - q_i|.

    Both arrays must be non-negative and sum to 1 (up to floating-point error).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * np.sum(np.abs(p - q))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap percentile confidence interval.

    Args:
        data:             1-D array of observations.
        statistic:        Function mapping 1-D array → scalar.
        n_resamples:      Number of bootstrap resamples.
        confidence_level: Coverage probability (default 0.95).
        rng:              Optional numpy random generator for reproducibility.

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(data)
    bootstrap_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(resample)
    alpha = 1.0 - confidence_level
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return statistic(data), ci_low, ci_high


def bootstrap_speedup_ci(
    baseline_tps: np.ndarray,
    spec_tps: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the speedup ratio, computed as ratio-of-means (not mean-of-ratios).

    The speedup is defined as:
        speedup = mean(spec_tps) / mean(baseline_tps)

    Resampling is done over prompts (paired resampling): for each resample,
    the same indices are drawn for both baseline and spec, preserving the
    pairing. This is required because prompt difficulty affects both conditions.

    Args:
        baseline_tps: tokens/sec for each prompt under baseline (shape: n_prompts).
        spec_tps:     tokens/sec for each prompt under speculative (shape: n_prompts).
        n_resamples:  Number of bootstrap resamples.
        confidence_level: Coverage probability.
        rng:          Optional numpy random generator.

    Returns:
        (speedup_point, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng()
    assert len(baseline_tps) == len(spec_tps), "Paired arrays must have equal length."
    n = len(baseline_tps)
    baseline_tps = np.asarray(baseline_tps, dtype=np.float64)
    spec_tps = np.asarray(spec_tps, dtype=np.float64)

    speedup_point = spec_tps.mean() / baseline_tps.mean()
    bootstrap_ratios = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_ratios[i] = spec_tps[idx].mean() / baseline_tps[idx].mean()

    alpha = 1.0 - confidence_level
    ci_low = np.percentile(bootstrap_ratios, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_ratios, 100 * (1 - alpha / 2))
    return speedup_point, ci_low, ci_high


def bootstrap_mean_ci(
    data: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of a 1-D array."""
    return bootstrap_ci(data, np.mean, n_resamples, confidence_level, rng)


# ---------------------------------------------------------------------------
# Chi-square goodness-of-fit
# ---------------------------------------------------------------------------

def chi_square_goodness_of_fit(
    observed_counts: np.ndarray,
    expected_probs: np.ndarray,
    min_expected: float = 5.0,
) -> Tuple[float, float]:
    """
    Chi-square goodness-of-fit test: does observed follow expected_probs?

    Cells with expected count < min_expected are merged to satisfy the
    standard assumption. Returns (chi2_statistic, p_value).

    Args:
        observed_counts: Integer counts per category.
        expected_probs:  Probabilities for each category (must sum to 1).
        min_expected:    Minimum expected count per cell before merging.
    """
    observed_counts = np.asarray(observed_counts, dtype=np.float64)
    expected_probs = np.asarray(expected_probs, dtype=np.float64)
    n = observed_counts.sum()
    expected_counts = expected_probs * n

    # Merge cells with expected count < min_expected.
    obs_merged = []
    exp_merged = []
    obs_acc = 0.0
    exp_acc = 0.0
    for o, e in zip(observed_counts, expected_counts):
        obs_acc += o
        exp_acc += e
        if exp_acc >= min_expected:
            obs_merged.append(obs_acc)
            exp_merged.append(exp_acc)
            obs_acc = 0.0
            exp_acc = 0.0
    if exp_acc > 0:
        obs_merged.append(obs_acc)
        exp_merged.append(exp_acc)

    if len(obs_merged) < 2:
        # Not enough categories for a meaningful test.
        return 0.0, 1.0

    obs_arr = np.array(obs_merged)
    exp_arr = np.array(exp_merged)
    chi2 = np.sum((obs_arr - exp_arr) ** 2 / exp_arr)
    dof = len(obs_arr) - 1
    p_value = 1.0 - ss.chi2.cdf(chi2, df=dof)
    return chi2, p_value


def chi_square_two_sample(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    min_expected: float = 5.0,
) -> Tuple[float, float]:
    """
    Two-sample chi-square test: do two multinomial samples come from the same distribution?

    Args:
        counts_a: Observed counts from distribution A, shape (vocab,).
        counts_b: Observed counts from distribution B, shape (vocab,).
        min_expected: Minimum expected count per cell.

    Returns:
        (chi2_statistic, p_value)
    """
    counts_a = np.asarray(counts_a, dtype=np.float64)
    counts_b = np.asarray(counts_b, dtype=np.float64)
    total_a = counts_a.sum()
    total_b = counts_b.sum()
    total = total_a + total_b

    # Keep only tokens that appear in either sample.
    mask = (counts_a + counts_b) > 0
    counts_a = counts_a[mask]
    counts_b = counts_b[mask]

    # Expected under pooled distribution.
    pooled = (counts_a + counts_b) / total
    expected_a = pooled * total_a
    expected_b = pooled * total_b

    # Merge cells.
    obs_pairs = list(zip(counts_a, counts_b, expected_a, expected_b))
    merged_obs_a, merged_obs_b, merged_exp_a, merged_exp_b = [], [], [], []
    acc_oa, acc_ob, acc_ea, acc_eb = 0, 0, 0, 0
    for oa, ob, ea, eb in obs_pairs:
        acc_oa += oa; acc_ob += ob; acc_ea += ea; acc_eb += eb
        if acc_ea >= min_expected and acc_eb >= min_expected:
            merged_obs_a.append(acc_oa)
            merged_obs_b.append(acc_ob)
            merged_exp_a.append(acc_ea)
            merged_exp_b.append(acc_eb)
            acc_oa = acc_ob = acc_ea = acc_eb = 0
    if acc_ea > 0:
        merged_obs_a.append(acc_oa); merged_obs_b.append(acc_ob)
        merged_exp_a.append(acc_ea); merged_exp_b.append(acc_eb)

    if len(merged_obs_a) < 2:
        return 0.0, 1.0

    chi2 = sum(
        (oa - ea) ** 2 / ea + (ob - eb) ** 2 / eb
        for oa, ob, ea, eb in zip(merged_obs_a, merged_obs_b, merged_exp_a, merged_exp_b)
    )
    dof = len(merged_obs_a) - 1
    p_value = 1.0 - ss.chi2.cdf(chi2, df=dof)
    return chi2, p_value


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------

def latency_percentiles(latencies_ms: np.ndarray) -> dict:
    """
    Compute p50, p90, p99 of inter-token latencies.

    Args:
        latencies_ms: 1-D array of inter-token latency values in milliseconds.

    Returns:
        dict with keys p50, p90, p99 (floats, milliseconds).
    """
    latencies_ms = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
    }
