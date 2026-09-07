"""
specbench/spec_decode.py — core speculative decoding loop.

Implements the Leviathan et al. (2023) / Chen et al. (2023) algorithm: a small
draft model proposes K tokens autoregressively; the target model verifies all K
proposals in a single forward pass; an accept/reject rule draws from the exact
target distribution.

Correctness invariants
----------------------
1. The output token distribution is identical to sampling from the target
   alone. This is guaranteed by the rejection-sampling maths (see
   compute_acceptance / sample_residual), not by implementation choice.
2. The same logits processors (temperature, top-k, top-p) are applied
   identically to q (draft) and p (target) before the ratio and the residual
   are computed. Warping only one side breaks the guarantee. See
   specbench/sampling.py.
3. Both KV caches are rolled back to the accepted prefix length after every
   round, including partial accepts. See specbench/kv.py.

Verification block layout
-------------------------
The naive formulation needs K+1 target distributions per round (p_1..p_{K+1})
but a K-token forward pass only yields K logit rows, and the missing one is
p_1 — the distribution over the first proposal, conditioned on the prefix
alone. Obtaining p_1 with a second, single-token target forward pass costs a
whole extra target call per round and roughly halves the achievable speedup.

Instead, this implementation carries a `pending` token: the token emitted at
the end of the previous round, which has been committed to the output but not
yet written into either KV cache. Each round feeds

    [pending, x_1, x_2, ..., x_K]        (K+1 positions)

to the target in a single forward pass. Output row j is the distribution over
the token that follows block position j, so

    row 0   -> p_1     (over x_1,   given prefix + pending)
    row i   -> p_{i+1} (over x_{i+1}, given prefix + pending + x_1..x_i)
    row K   -> p_{K+1} (the bonus token, given all K proposals accepted)

That is exactly the K+1 distributions the accept/reject rule needs, from one
target forward pass per round. Cost per round is therefore 1 target call plus
K (occasionally K+1) draft calls, which is what makes the method profitable.

Cache bookkeeping that follows from this: entering a round both caches hold
`committed_len` positions and exclude `pending`. The target block appends
K+1 positions; after accepting `a` of the K proposals the committed prefix is
`committed_len + 1 + a` (the +1 is `pending` itself, which is always correct
by construction). The draft only needs to be fed pending + x_1..x_{K-1} to
produce all K proposals, so it is one position short when a == K; that single
extra draft call is made only in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from specbench.kv import (
    DynamicCache,
    cache_seq_len,
    make_empty_cache,
    multi_token_forward,
    prefill_cache,
    rollback_cache,
    single_token_forward,
)
from specbench.sampling import (
    greedy_token,
    logits_to_probs,
    make_generator,
    sample_token,
    uniform_random,
)

# Hard cap on rounds, as a runaway guard. Each round emits at least one token,
# so max_new_tokens rounds always suffice.
_ROUND_SAFETY_FACTOR = 1


@dataclass
class RoundResult:
    """
    Outcome of one speculative decoding round.

    accepted_count:    number of draft proposals accepted this round.
    total_emitted:     tokens emitted this round (accepted + 1 resampled/bonus).
    tokens_emitted:    the emitted token ids.
    residual_fallback: True if the residual distribution was degenerate and p
                       was sampled directly instead.
    """

    accepted_count: int
    total_emitted: int
    tokens_emitted: List[int]
    residual_fallback: bool = False


# ---------------------------------------------------------------------------
# Accept/reject logic (pure — no model calls, no cache)
# ---------------------------------------------------------------------------

def compute_acceptance(
    p_i: torch.Tensor,
    q_i: torch.Tensor,
    x_i: int,
    generator: Optional[torch.Generator] = None,
) -> bool:
    """
    Accept or reject draft token x_i under the Leviathan et al. rule.

    p_i: target distribution over the vocabulary, shape (vocab,).
    q_i: draft distribution over the vocabulary, shape (vocab,).
    x_i: the proposed token index.

        r ~ Uniform(0, 1)
        accept iff r < min(1, p_i[x_i] / q_i[x_i])

    Evaluated in float32: FP16 probabilities lose enough precision in the tail
    that the ratio can be visibly wrong for rare tokens.
    """
    p_val = p_i[x_i].float().item()
    q_val = q_i[x_i].float().item()
    if q_val == 0.0:
        # The draft could not have proposed this token under its own warped
        # distribution; treat as a rejection rather than dividing by zero.
        return False
    threshold = min(1.0, p_val / q_val)
    r = uniform_random(generator=generator, device=str(p_i.device))
    return r < threshold


def sample_residual(
    p_i: torch.Tensor,
    q_i: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> tuple[int, bool]:
    """
    Sample from the residual distribution normalize(max(0, p_i - q_i)).

    Returns (token_id, used_fallback). used_fallback is True when the residual
    summed to ~0, in which case p_i is sampled directly. That happens only when
    q_i >= p_i everywhere, which given sum(p) = sum(q) = 1 means q == p, so the
    token was accepted with probability 1 and this branch should be unreachable
    in exact arithmetic. A nonzero and growing fallback count therefore signals
    a real numerical problem, and is recorded in every run record.
    """
    p_i = p_i.float()
    q_i = q_i.float()
    residual = torch.clamp(p_i - q_i, min=0.0)
    residual_sum = residual.sum().item()

    if residual_sum < 1e-8:
        token = int(sample_token(p_i.unsqueeze(0), generator=generator).item())
        return token, True

    residual = residual / residual_sum
    token = int(sample_token(residual.unsqueeze(0), generator=generator).item())
    return token, False


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_speculative(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
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
    Generate text with speculative decoding.

    One round:
      1. The draft proposes K tokens autoregressively.
      2. The target verifies all K in ONE forward pass over
         [pending, x_1..x_K] (see the module docstring).
      3. Proposals are accepted/rejected in order by the rejection rule.
      4. All K accepted -> emit K+1 tokens (the last from the bonus
         distribution p_{K+1}). Otherwise emit a accepted tokens plus one
         token drawn from the residual at the rejection point.
      5. Both caches roll back to the accepted prefix length.

    Distribution guarantee: the marginal distribution of every emitted token
    equals the target model's distribution under the same sampling parameters.
    Verified empirically in tests/test_sampler_synthetic.py.

    Args:
        draft_model:    Small model producing proposals.
        target_model:   Large model performing verification.
        input_ids:      Prompt token ids, shape (1, prompt_len).
        max_new_tokens: Stop after this many new tokens.
        K:              Proposals per round.
        temperature:    Sampling temperature (0 = greedy).
        top_k:          Top-k filtering (0 = off).
        top_p:          Nucleus filtering (1.0 = off).
        device:         Device string.
        seed:           RNG seed. If None the run is not reproducible and
                        `reproducible` in the result is False.
        eos_token_id:   Stop once this token is emitted.
        clock:          Optional monotonic clock (e.g. time.perf_counter). When
                        given, a timestamp is recorded at the end of every
                        round, giving real token arrival times. See
                        specbench/harness.py for how these are turned into
                        latency percentiles.

    Returns:
        dict with generated_ids, accepted_per_round, target_forward_calls,
        draft_forward_calls, residual_fallback_count, round_end_times,
        tokens_per_round, reproducible.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape (1, prompt_len), got {tuple(input_ids.shape)}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    generator = make_generator(device, seed)
    greedy = temperature == 0.0
    input_ids = input_ids.to(device)

    # `pending` is the most recently emitted token: committed to the output,
    # but deliberately not yet written into either KV cache, because it is fed
    # at the head of the next verification block. Round 1 seeds it with the
    # last prompt token, so the prefill covers input_ids[:, :-1] only.
    prefix_ids = input_ids[:, :-1]
    pending_token = int(input_ids[0, -1].item())

    draft_cache = make_empty_cache()
    target_cache = make_empty_cache()
    target_forward_calls = 0
    draft_forward_calls = 0

    if prefix_ids.shape[1] > 0:
        _, draft_cache = prefill_cache(draft_model, prefix_ids, draft_cache)
        _, target_cache = prefill_cache(target_model, prefix_ids, target_cache)
        draft_forward_calls += 1
        target_forward_calls += 1

    committed_len = prefix_ids.shape[1]

    generated_ids: List[int] = []
    accepted_per_round: List[int] = []
    tokens_per_round: List[int] = []
    round_end_times: List[float] = []
    residual_fallback_count = 0

    t_start = clock() if clock is not None else None
    max_rounds = max_new_tokens * _ROUND_SAFETY_FACTOR + 1
    stop = False

    for _round in range(max_rounds):
        if stop or len(generated_ids) >= max_new_tokens:
            break

        # ------------------------------------------------------------------
        # Step 1: draft K proposals, starting from `pending`.
        #
        # Feeding pending here (rather than at the end of the previous round)
        # keeps the draft cache aligned with the target's committed prefix and
        # costs the same number of draft calls.
        # ------------------------------------------------------------------
        proposed_tokens: List[int] = []
        q_distributions: List[torch.Tensor] = []
        current_input = pending_token

        for _step in range(K):
            token_tensor = torch.tensor([[current_input]], dtype=torch.long, device=device)
            draft_logits, draft_cache = single_token_forward(
                draft_model, token_tensor, draft_cache
            )
            draft_forward_calls += 1
            logits_1d = draft_logits[0, -1, :]

            q_i = logits_to_probs(logits_1d, temperature, top_k, top_p)
            if greedy:
                x_i = int(greedy_token(logits_1d).item())
            else:
                x_i = int(sample_token(q_i.unsqueeze(0), generator=generator).item())

            proposed_tokens.append(x_i)
            q_distributions.append(q_i)
            current_input = x_i

        # Draft cache now holds committed_len + K positions:
        # pending, x_1, ..., x_{K-1}. x_K is deliberately not fed yet.

        # ------------------------------------------------------------------
        # Step 2: verify — ONE target forward pass over [pending, x_1..x_K].
        # Row j of the output is the distribution over the token following
        # block position j, so rows 0..K are exactly p_1..p_{K+1}.
        # ------------------------------------------------------------------
        block = torch.tensor(
            [[pending_token] + proposed_tokens], dtype=torch.long, device=device
        )  # (1, K+1)
        target_logits_block, target_cache = multi_token_forward(
            target_model, block, target_cache
        )
        target_forward_calls += 1

        p_distributions: List[torch.Tensor] = [
            logits_to_probs(target_logits_block[0, j, :], temperature, top_k, top_p)
            for j in range(K + 1)
        ]

        # ------------------------------------------------------------------
        # Step 3: accept/reject in order.
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
                # The greedy residual collapses to the target's argmax.
                x_prime = int(greedy_token(p_i).item())
            else:
                x_prime, fell_back = sample_residual(p_i, q_i, generator=generator)
                if fell_back:
                    residual_fallback_count += 1
            emitted_tokens.append(x_prime)
            break
        else:
            # Every proposal accepted: the verification pass already gave us
            # p_{K+1} for free, so a bonus token comes at no extra cost. This
            # is why the maximum tokens per round is K+1, not K.
            p_bonus = p_distributions[K]
            if greedy:
                x_bonus = int(greedy_token(p_bonus).item())
            else:
                x_bonus = int(sample_token(p_bonus.unsqueeze(0), generator=generator).item())
            emitted_tokens.append(x_bonus)
            accepted_count = K

        # ------------------------------------------------------------------
        # Step 4: roll both caches back to the accepted prefix.
        #
        # `pending` is always correct (it was emitted, not proposed), so the
        # committed prefix grows by 1 + accepted_count. The final emitted token
        # becomes the next round's `pending` and stays out of both caches.
        # ------------------------------------------------------------------
        new_committed_len = committed_len + 1 + accepted_count

        if accepted_count == K:
            # x_K was never fed to the draft; feed it so the draft cache can
            # reach the same committed length as the target's.
            last_proposal = torch.tensor(
                [[proposed_tokens[-1]]], dtype=torch.long, device=device
            )
            _, draft_cache = single_token_forward(draft_model, last_proposal, draft_cache)
            draft_forward_calls += 1

        rollback_cache(draft_cache, new_committed_len)
        rollback_cache(target_cache, new_committed_len)
        committed_len = new_committed_len
        pending_token = emitted_tokens[-1]

        # ------------------------------------------------------------------
        # Step 5: record. Truncate at EOS so that the token stream matches what
        # the baseline would have produced (the baseline stops the instant it
        # emits EOS; without this, a round could emit tokens past it).
        # ------------------------------------------------------------------
        if eos_token_id is not None and eos_token_id in emitted_tokens:
            cut = emitted_tokens.index(eos_token_id) + 1
            emitted_tokens = emitted_tokens[:cut]
            stop = True

        generated_ids.extend(emitted_tokens)
        accepted_per_round.append(accepted_count)
        tokens_per_round.append(len(emitted_tokens))

        if clock is not None:
            # Every emitted token became available at this instant: the round's
            # accept/reject decisions all followed the same verification pass.
            # The .item() calls above already synchronised the device, so this
            # timestamp does not need (and must not add) an extra sync.
            round_end_times.append(clock() - t_start)

    generated_ids = generated_ids[:max_new_tokens]

    return {
        "generated_ids": generated_ids,
        "accepted_per_round": accepted_per_round,
        "tokens_per_round": tokens_per_round,
        "round_end_times_s": round_end_times,
        "target_forward_calls": target_forward_calls,
        "draft_forward_calls": draft_forward_calls,
        "residual_fallback_count": residual_fallback_count,
        "reproducible": seed is not None,
    }
