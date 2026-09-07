"""
specbench/eagle/spec_decode_eagle.py — speculative decoding driven by the EAGLE head.

The accept/reject rule is identical to Phase 1 (see specbench/spec_decode.py).
Only the source of the draft distribution q changes: instead of a separate
small model with its own KV cache, an EAGLE-style head autoregresses on the
target model's own last-layer features.

The distribution guarantee is therefore unchanged. A bad draft head costs
acceptance rate — never correctness.

Feature/token pairing
---------------------
The head is trained (see eagle/train.py) on pairs

    head(f_{t-1}, embed(token_t))  ->  f_t

where f_t is the target's last-layer feature at position t, and the target's
frozen LM head maps f_t to the distribution over token t+1. Inference must use
the same pairing or acceptance collapses: the head would be fed a feature one
position later than anything it saw in training.

So a round starts from (f_{t-1}, token_t) where token_t is the pending token,
and autoregresses by substituting the head's own predicted features:

    step 1: head(f_{t-1},   e(token_t)) -> f̂_t   -> q over x_1;  sample x_1
    step 2: head(f̂_t,       e(x_1))     -> f̂_{t+1} -> q over x_2;  sample x_2
    ...

Verification block
------------------
As in Phase 1, the block is [pending, x_1..x_K] so that one target forward
pass yields p_1..p_{K+1}. Requesting hidden states from that same pass gives
the real features at every block position, which is what makes the single-pass
form work here: after accepting `a` proposals, the feature needed to seed the
next round is f at block index `a` — the predecessor of the newly emitted
token — and it is already in hand. No second target pass is required.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from specbench.kv import (
    DynamicCache,
    cache_seq_len,
    make_empty_cache,
    rollback_cache,
)
from specbench.sampling import (
    greedy_token,
    logits_to_probs,
    make_generator,
    sample_token,
)
from specbench.spec_decode import compute_acceptance, sample_residual


# Bumped whenever a change to this file invalidates previously measured EAGLE
# results. scripts/run_phase2_sweep.py records it in each per-head summary and
# re-measures any head whose recorded version differs, so a fix to the decoding
# path cannot be masked by the "summary already exists" skip.
#
#   1 — initial version.
#   2 — the head carries its own KV cache across draft steps, and is seeded over
#       the prompt. Version 1 called the head on length-1 sequences with no
#       history, a computation it was never trained for; measured acceptance
#       0.231 against 0.279 in the training regime.
INFERENCE_VERSION = 2


def _target_forward(
    target_model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: Optional[DynamicCache],
):
    """Forward pass that also returns last-layer hidden states."""
    with torch.no_grad():
        out = target_model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
    return out


def generate_eagle(
    target_model: torch.nn.Module,
    eagle_head: torch.nn.Module,
    token_embedding: torch.nn.Embedding,
    lm_head_weight: torch.Tensor,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    K: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = "cuda",
    seed: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    clock: Optional[Callable[[], float]] = None,
) -> dict:
    """
    Generate text using EAGLE-style speculative decoding.

    One target forward pass per round; K head forward passes per round. The
    head is roughly two orders of magnitude cheaper than the target, so the
    round cost is dominated by the single verification pass — the same
    economics as Phase 1, minus the draft model's separate weights and KV
    cache in VRAM.

    Returns:
        dict with generated_ids, accepted_per_round, tokens_per_round,
        round_end_times_s, target_forward_calls, draft_forward_calls (head
        calls), residual_fallback_count, reproducible.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape (1, prompt_len), got {tuple(input_ids.shape)}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    generator = make_generator(device, seed)
    greedy = temperature == 0.0
    input_ids = input_ids.to(device)
    eagle_head.eval()

    # As in Phase 1, the last prompt token is held back as `pending` and fed at
    # the head of the first verification block.
    prefix_ids = input_ids[:, :-1]
    pending_token = int(input_ids[0, -1].item())

    target_cache = make_empty_cache()
    # The head is an autoregressive decoder and needs its own KV cache, kept in
    # step with the target's committed prefix and rolled back alongside it. It
    # holds one entry per committed token, written when that token is first fed
    # to the head as `pending`.
    head_cache = make_empty_cache()
    target_forward_calls = 0
    head_forward_calls = 0

    if prefix_ids.shape[1] > 0:
        out = _target_forward(target_model, prefix_ids, target_cache)
        target_cache = out.past_key_values
        prefix_hidden = out.hidden_states[-1]  # (1, P, hidden)
        # Predecessor feature of `pending`, i.e. f_{t-1} for the pairing above.
        prev_feature = prefix_hidden[0, -1, :]
        target_forward_calls += 1

        # Seed the head's cache over the prompt, in one batched call, so that
        # its cache length equals committed_len from the first round onwards.
        # Without this the head's positions drift out of step with the target's
        # and the rollback arithmetic silently addresses the wrong entries.
        # It also gives the head the prompt as attention context, which is the
        # regime it was trained in.
        P = prefix_ids.shape[1]
        hidden_size = prefix_hidden.shape[-1]
        with torch.no_grad():
            prev_seq = torch.cat(
                [
                    torch.zeros(1, 1, hidden_size, dtype=torch.float32, device=device),
                    prefix_hidden[:, : P - 1, :].float(),
                ],
                dim=1,
            )  # (1, P, hidden) — f_{i-1}, with zeros at i=0 as in training
            tok_seq = token_embedding(prefix_ids).float()  # (1, P, hidden) — e(t_i)
            eagle_head(prev_seq, tok_seq, past_key_values=head_cache, use_cache=True)
        head_forward_calls += 1
    else:
        # Single-token prompt: there is no predecessor feature. Zeros are the
        # same thing the training data uses for the first position of a chunk.
        hidden_size = int(token_embedding.weight.shape[1])
        prev_feature = torch.zeros(hidden_size, dtype=torch.float32, device=device)

    committed_len = prefix_ids.shape[1]

    generated_ids: List[int] = []
    accepted_per_round: List[int] = []
    tokens_per_round: List[int] = []
    round_end_times: List[float] = []
    residual_fallback_count = 0

    t_start = clock() if clock is not None else None
    stop = False

    for _round in range(max_new_tokens + 1):
        if stop or len(generated_ids) >= max_new_tokens:
            break

        # ------------------------------------------------------------------
        # Step 1: the head proposes K tokens.
        # ------------------------------------------------------------------
        proposed_tokens: List[int] = []
        q_distributions: List[torch.Tensor] = []

        feature = prev_feature
        current_token = pending_token

        # The head's cache is rolled back to `committed_len` at the end of each
        # round, so it starts this round holding exactly the committed prefix.
        # Each of the K draft steps appends one entry; the first (for `pending`)
        # is kept, the speculative remainder is discarded in the rollback below.
        for _step in range(K):
            with torch.no_grad():
                tok_emb = token_embedding(
                    torch.tensor([[current_token]], dtype=torch.long, device=device)
                ).float()  # (1, 1, hidden)
                predicted = eagle_head(
                    feature.float().view(1, 1, -1),
                    tok_emb,
                    past_key_values=head_cache,
                    use_cache=True,
                )  # (1, 1, hidden)
                predicted_1d = predicted[0, 0, :]
                # Match the LM head's dtype rather than assuming FP16: the
                # target is FP16 on the GPU but FP32 in CPU tests, and a
                # hard-coded .half() makes the two paths diverge.
                logits_1d = (
                    predicted_1d.to(lm_head_weight.dtype) @ lm_head_weight.t()
                ).float()
            head_forward_calls += 1

            q_i = logits_to_probs(logits_1d, temperature, top_k, top_p)
            if greedy:
                x_i = int(greedy_token(logits_1d).item())
            else:
                x_i = int(sample_token(q_i.unsqueeze(0), generator=generator).item())

            proposed_tokens.append(x_i)
            q_distributions.append(q_i)
            feature = predicted_1d
            current_token = x_i

        # ------------------------------------------------------------------
        # Step 2: verify — ONE target forward pass over [pending, x_1..x_K].
        # ------------------------------------------------------------------
        block = torch.tensor(
            [[pending_token] + proposed_tokens], dtype=torch.long, device=device
        )
        out = _target_forward(target_model, block, target_cache)
        target_cache = out.past_key_values
        target_logits_block = out.logits              # (1, K+1, vocab)
        target_hidden_block = out.hidden_states[-1]   # (1, K+1, hidden)
        target_forward_calls += 1

        p_distributions = [
            logits_to_probs(target_logits_block[0, j, :], temperature, top_k, top_p)
            for j in range(K + 1)
        ]

        # ------------------------------------------------------------------
        # Step 3: accept/reject — identical rule to Phase 1.
        # ------------------------------------------------------------------
        accepted_count = 0
        emitted_tokens: List[int] = []

        for i in range(K):
            x_i = proposed_tokens[i]
            p_i = p_distributions[i]
            q_i = q_distributions[i]

            if greedy:
                accept = int(p_i.argmax().item()) == x_i
            else:
                accept = compute_acceptance(p_i, q_i, x_i, generator=generator)

            if accept:
                emitted_tokens.append(x_i)
                accepted_count += 1
                continue

            if greedy:
                x_prime = int(greedy_token(p_i).item())
            else:
                x_prime, fell_back = sample_residual(p_i, q_i, generator=generator)
                if fell_back:
                    residual_fallback_count += 1
            emitted_tokens.append(x_prime)
            break
        else:
            p_bonus = p_distributions[K]
            if greedy:
                x_bonus = int(greedy_token(p_bonus).item())
            else:
                x_bonus = int(sample_token(p_bonus.unsqueeze(0), generator=generator).item())
            emitted_tokens.append(x_bonus)
            accepted_count = K

        # ------------------------------------------------------------------
        # Step 4: roll back and seed the next round.
        #
        # Block index j holds the feature at absolute position committed_len+j.
        # The newly emitted token will sit at committed_len + 1 + accepted,
        # so its predecessor feature is block index `accepted_count`.
        # ------------------------------------------------------------------
        new_committed_len = committed_len + 1 + accepted_count
        rollback_cache(target_cache, new_committed_len)

        # The head's cache grew by K during drafting: one entry for `pending`
        # (absolute position committed_len) and K-1 for x_1..x_{K-1}. The
        # committed prefix needs entries through x_{accepted_count}, i.e. length
        # committed_len + 1 + accepted_count. That is within what we have unless
        # every proposal was accepted, in which case x_K was never fed to the
        # head and the cache is one entry short — the same situation as the
        # draft model in Phase 1, and handled the same way.
        if accepted_count == K:
            with torch.no_grad():
                tok_emb = token_embedding(
                    torch.tensor([[current_token]], dtype=torch.long, device=device)
                ).float()
                eagle_head(
                    feature.float().view(1, 1, -1),
                    tok_emb,
                    past_key_values=head_cache,
                    use_cache=True,
                )
            head_forward_calls += 1

        # Entries for rejected proposals must go, for exactly the reason the
        # target's must: otherwise the head attends to tokens never emitted.
        rollback_cache(head_cache, new_committed_len)

        prev_feature = target_hidden_block[0, accepted_count, :]
        committed_len = new_committed_len
        pending_token = emitted_tokens[-1]

        if eos_token_id is not None and eos_token_id in emitted_tokens:
            cut = emitted_tokens.index(eos_token_id) + 1
            emitted_tokens = emitted_tokens[:cut]
            stop = True

        generated_ids.extend(emitted_tokens)
        accepted_per_round.append(accepted_count)
        tokens_per_round.append(len(emitted_tokens))
        if clock is not None:
            round_end_times.append(clock() - t_start)

    generated_ids = generated_ids[:max_new_tokens]

    return {
        "generated_ids": generated_ids,
        "accepted_per_round": accepted_per_round,
        "tokens_per_round": tokens_per_round,
        "round_end_times_s": round_end_times,
        "target_forward_calls": target_forward_calls,
        "draft_forward_calls": head_forward_calls,
        "residual_fallback_count": residual_fallback_count,
        "reproducible": seed is not None,
        # Stamped into every run record so that measurements taken with an
        # older decoding path can be identified and excluded from aggregation
        # rather than silently averaged in with current ones.
        "inference_version": INFERENCE_VERSION,
    }
