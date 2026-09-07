"""
specbench/kv.py — KV cache handling and rollback.

The KV cache in HuggingFace transformers is a list of (key, value) tensor
tuples, one per layer. Each tensor has shape:
    (batch, num_kv_heads, seq_len, head_dim)

Rolling back to an accepted prefix of length L means truncating dim=2 of
every K and V tensor to L. Both draft and target caches must be rolled back
independently after each verification round.

Correctness note: getting the rollback wrong produces subtly corrupted output
that still looks plausible. test_kv_rollback.py exists specifically to catch
this class of error.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import DynamicCache


# ---------------------------------------------------------------------------
# Cache creation
# ---------------------------------------------------------------------------

def make_empty_cache() -> DynamicCache:
    """Return a fresh DynamicCache for use with a HuggingFace model."""
    return DynamicCache()


def cache_seq_len(cache: DynamicCache) -> int:
    """
    Return the current sequence length stored in the KV cache.
    Returns 0 if the cache is empty.
    """
    if len(cache.key_cache) == 0:
        return 0
    # key_cache[0] has shape (batch, heads, seq_len, head_dim)
    return cache.key_cache[0].shape[2]


# ---------------------------------------------------------------------------
# Cache rollback
# ---------------------------------------------------------------------------

def rollback_cache(cache: DynamicCache, keep_len: int) -> DynamicCache:
    """
    Truncate the KV cache to the first `keep_len` token positions.

    This is called after a rejection at position i (0-indexed), where
    keep_len = i (the number of accepted tokens before the rejection).

    Both draft and target caches must be rolled back before the next round.

    Args:
        cache:    The DynamicCache to truncate in-place.
        keep_len: Number of token positions to keep (0-indexed end, exclusive).
    Returns:
        The same DynamicCache object, truncated in place.
    """
    if len(cache.key_cache) == 0:
        return cache

    for layer_idx in range(len(cache.key_cache)):
        cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :keep_len, :]
        cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :keep_len, :]

    return cache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """
    Return a deep copy of the KV cache. Used in tests that need to
    compare generation from a rolled-back cache vs. a freshly-built one.
    """
    new_cache = DynamicCache()
    for layer_idx in range(len(cache.key_cache)):
        new_cache.key_cache.append(cache.key_cache[layer_idx].clone())
        new_cache.value_cache.append(cache.value_cache[layer_idx].clone())
    return new_cache


# ---------------------------------------------------------------------------
# Prefill helpers
# ---------------------------------------------------------------------------

def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: Optional[DynamicCache] = None,
) -> tuple[torch.Tensor, DynamicCache]:
    """
    Run a forward pass over `input_ids`, populating the KV cache.

    Args:
        model:     The language model (target or draft).
        input_ids: shape (1, seq_len). Must be on the same device as the model.
        cache:     Optional existing DynamicCache; created fresh if None.
    Returns:
        (logits, cache) where logits has shape (1, seq_len, vocab_size).
    """
    if cache is None:
        cache = make_empty_cache()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
    return out.logits, out.past_key_values


def single_token_forward(
    model: torch.nn.Module,
    token_id: torch.Tensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, DynamicCache]:
    """
    Run a single-token forward pass, extending the KV cache by one position.

    Args:
        model:    The language model.
        token_id: shape (1, 1), the new token to process.
        cache:    Existing populated cache (will be extended).
    Returns:
        (logits, cache) where logits has shape (1, 1, vocab_size).
    """
    with torch.no_grad():
        out = model(
            input_ids=token_id,
            past_key_values=cache,
            use_cache=True,
        )
    return out.logits, out.past_key_values


def multi_token_forward(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, DynamicCache]:
    """
    Run a forward pass over a block of K tokens.

    Used by the target model verification step: the K proposed tokens are fed
    together, yielding K logit rows (positions 0..K-1 of the new block), which
    correspond to distributions p_1 .. p_K. The K-th output (index K-1) is
    p_{K+1} conditioned on the full accepted prefix plus all K proposals.

    Args:
        model:     The target model.
        token_ids: shape (1, K), the proposed tokens.
        cache:     Existing cache containing the verified prefix so far.
    Returns:
        (logits, cache) where logits has shape (1, K, vocab_size).
    """
    with torch.no_grad():
        out = model(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
        )
    return out.logits, out.past_key_values
