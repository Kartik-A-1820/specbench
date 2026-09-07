"""
specbench/sampling.py — logits processors for temperature, top-k, and top-p.

These processors are shared between the draft and target model paths.
Applying the same transformation to both distributions before computing the
accept/reject ratio is required for distribution preservation — warping only
one side breaks the theoretical guarantee.

Usage pattern:
    logits = apply_logits_processors(logits, temperature, top_k, top_p)
    probs  = torch.softmax(logits, dim=-1)

All operations are done in float32 to avoid precision loss in softmax and
probability arithmetic. Models may return float16 logits; callers should
cast before passing here, or the functions will cast internally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature.

    temperature = 0  → greedy (handled separately; passing 0 here raises).
    temperature = 1  → no change.
    temperature > 1  → softer distribution.
    temperature < 1  → sharper distribution.
    """
    if temperature == 0.0:
        raise ValueError(
            "temperature=0 is greedy; use greedy_sample() or check callers."
        )
    return logits.float() / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k, keeping only the k highest values.

    top_k = 0 means no filtering.
    """
    if top_k <= 0:
        return logits
    logits = logits.float()
    top_k = min(top_k, logits.size(-1))
    # kth_val is the k-th largest; anything strictly below it gets -inf.
    kth_val = logits.topk(top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_val, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering. Retain the minimal set of tokens whose
    cumulative probability mass is >= top_p; set others to -inf.

    top_p = 1.0 means no filtering.
    """
    if top_p >= 1.0:
        return logits
    logits = logits.float()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once their cumulative mass exceeds top_p.
    # Shift by one so that the first token exceeding the threshold is kept.
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

    # Scatter back to original ordering.
    logits = torch.zeros_like(logits).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )
    return logits


def apply_logits_processors(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Apply the full sampling pipeline: temperature → top-k → top-p.

    Returns modified logits (float32). For greedy decoding (temperature=0),
    use greedy_token() instead.

    This function must be called identically for both draft (q) and target (p)
    distributions before computing the accept/reject ratio. Asymmetric warping
    would break distribution preservation.
    """
    if temperature == 0.0:
        # Caller should use greedy_token(); return unmodified.
        return logits.float()
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return logits


def make_generator(device: str, seed: Optional[int]) -> Optional[torch.Generator]:
    """
    Build a torch.Generator on `device`, seeded with `seed`.

    Returns None when seed is None, in which case callers fall back to the
    global RNG and the run is NOT reproducible.

    The generator must live on the same device as the tensors it will sample
    from: torch.multinomial and torch.rand both reject a device mismatch.
    Creating it here (rather than always on CPU) is what makes `seed` mean
    something on CUDA — see sample_token().
    """
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def sample_token(
    probs: torch.Tensor,
    generator: "Optional[torch.Generator]" = None,
) -> torch.Tensor:
    """
    Sample one token from a probability distribution.

    Args:
        probs:     shape (..., vocab_size), already normalised. float32.
        generator: Optional torch.Generator for reproducible sampling. It must
                   be on the same device as `probs`; a mismatch raises rather
                   than silently falling back to the global RNG, which would
                   make a recorded `seed` a lie.
    Returns:
        token id tensor, shape (...).
    """
    flat = probs.view(-1, probs.size(-1))
    if generator is None:
        return torch.multinomial(flat, num_samples=1).squeeze(-1)
    if generator.device.type != flat.device.type:
        raise ValueError(
            f"Generator is on {generator.device.type} but probs are on "
            f"{flat.device.type}. Sampling would silently ignore the generator "
            "and the run would not be reproducible. Build the generator with "
            "make_generator(device, seed)."
        )
    return torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)


def uniform_random(
    generator: "Optional[torch.Generator]" = None,
    device: str = "cpu",
) -> float:
    """
    Draw a single Uniform(0, 1) value, honouring `generator` if given.

    Used by the accept/reject rule. Kept here so that every source of
    randomness in the project goes through one seeded path.
    """
    if generator is None:
        return float(torch.rand(1, device=device).item())
    return float(torch.rand(1, device=generator.device, generator=generator).item())


def greedy_token(logits: torch.Tensor) -> torch.Tensor:
    """
    Return the argmax token (greedy decoding). Works on float16 or float32.
    """
    return logits.argmax(dim=-1)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Convert raw logits to a normalised probability distribution.

    Applies logits processors and returns softmax probabilities in float32.
    This is the canonical path for obtaining q and p distributions.
    """
    if temperature == 0.0:
        # Greedy: one-hot distribution at the argmax.
        vocab_size = logits.size(-1)
        idx = logits.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, idx, 1.0)
        return probs
    processed = apply_logits_processors(logits, temperature, top_k, top_p)
    return F.softmax(processed, dim=-1)
