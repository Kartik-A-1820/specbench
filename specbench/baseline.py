"""
specbench/baseline.py — target-only autoregressive reference implementation.

The baseline uses the same KV cache handling and logits processors as the
speculative path. This is essential: a baseline with a different (especially
simpler or accidentally faster) implementation would produce a speedup number
that reflects an artefact rather than speculation.

The baseline serves two purposes:
  1. As the reference for correctness tests (test_greedy_equivalence.py).
  2. As the denominator in the wall-clock speedup ratio.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from specbench.kv import DynamicCache, make_empty_cache, single_token_forward
from specbench.sampling import (
    greedy_token,
    logits_to_probs,
    make_generator,
    sample_token,
)


def generate_baseline(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = "cuda",
    seed: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    clock: Optional[Callable[[], float]] = None,
) -> dict:
    """
    Generate tokens autoregressively using the target model only.

    Args:
        model:          The target language model.
        input_ids:      Prompt token IDs, shape (1, prompt_len).
        max_new_tokens: Stop after this many new tokens.
        temperature:    Sampling temperature (0 = greedy).
        top_k:          Top-k filtering.
        top_p:          Nucleus filtering.
        device:         Device string.
        seed:           RNG seed. If None the run is not reproducible.
        eos_token_id:   Stop if this token is generated.
        clock:          Optional monotonic clock. When given, a timestamp is
                        recorded per emitted token, giving real (not inferred)
                        inter-token latencies. One token per round here, so the
                        instrumentation is identical in kind to the speculative
                        path's per-round timestamps.

    Returns:
        dict with keys:
          generated_ids:        list of generated token ids.
          target_forward_calls: total forward passes (= generated_tokens + 1 prefill).
          round_end_times_s:    per-token arrival times, seconds from start.
          tokens_per_round:     all ones; present so both paths share a schema.
    """
    generator = make_generator(device, seed)

    greedy = temperature == 0.0
    input_ids = input_ids.to(device)

    # Prefill.
    cache = make_empty_cache()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
    cache = out.past_key_values
    last_logit = out.logits[0, -1, :]
    target_forward_calls = 1

    generated_ids: List[int] = []
    round_end_times: List[float] = []
    t_start = clock() if clock is not None else None

    for _ in range(max_new_tokens):
        # Sample the next token.
        if greedy:
            token_id = int(greedy_token(last_logit).item())
        else:
            probs = logits_to_probs(last_logit, temperature, top_k, top_p)
            token_id = int(sample_token(probs.unsqueeze(0), generator=generator).item())

        generated_ids.append(token_id)
        if clock is not None:
            # Taken straight after the .item() above, which has already
            # synchronised the device. No extra sync is inserted, so the
            # instrumentation does not bias the baseline against the
            # speculative path.
            round_end_times.append(clock() - t_start)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        # Feed the generated token to advance the cache.
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        new_logits, cache = single_token_forward(model, token_tensor, cache)
        last_logit = new_logits[0, 0, :]
        target_forward_calls += 1

    return {
        "generated_ids": generated_ids,
        "target_forward_calls": target_forward_calls,
        "draft_forward_calls": 0,
        "round_end_times_s": round_end_times,
        "tokens_per_round": [1] * len(generated_ids),
        "accepted_per_round": [],
        "residual_fallback_count": 0,
        "reproducible": seed is not None,
    }
