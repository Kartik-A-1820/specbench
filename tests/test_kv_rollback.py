"""
tests/test_kv_rollback.py — KV cache rollback correctness.

Rollback is the failure mode that does not announce itself. If the cache keeps
stale key/value entries for tokens that were rejected, attention still runs,
logits still look like logits, and the model still emits fluent text — it is
just conditioned on a prefix that was never emitted. Nothing crashes and the
output is plausible, so only a test that forces divergence can catch it.

The central test here therefore does three things that a weaker version would
not:

  1. It extends the cache with tokens that are DIFFERENT from the ones it will
     later regenerate. If the poisoned branch happened to be the same tokens,
     a broken rollback and a correct one give identical output and the test
     proves nothing.
  2. It checks that the poisoned branch actually changes the next-token
     prediction, so we know the test has teeth on this model and prompt.
  3. It drives the real speculative decoder with a deliberately hostile draft
     model, so that rejections genuinely occur, and compares against the
     baseline.

Requires CUDA.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from specbench.baseline import generate_baseline
from specbench.kv import (
    cache_seq_len,
    clone_cache,
    make_empty_cache,
    prefill_cache,
    rollback_cache,
    single_token_forward,
)
from specbench.models import load_draft_model, load_target_model, verify_cuda_environment
from specbench.sampling import greedy_token
from specbench.spec_decode import generate_speculative

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available.",
)

MODEL_DIR = Path(__file__).parent.parent / "models"
DEVICE = "cuda"


@pytest.fixture(scope="module")
def target_model_and_tokenizer():
    verify_cuda_environment()
    model, tokenizer = load_target_model(MODEL_DIR, device=DEVICE)
    return model, tokenizer


def _greedy_continue(model, last_logit, cache, n):
    """Greedily emit n tokens from (last_logit, cache). Mutates cache."""
    out = []
    for _ in range(n):
        tok = int(greedy_token(last_logit).item())
        out.append(tok)
        tok_t = torch.tensor([[tok]], dtype=torch.long, device=DEVICE)
        new_logits, cache = single_token_forward(model, tok_t, cache)
        last_logit = new_logits[0, 0, :]
    return out, cache


def test_rollback_cache_length(target_model_and_tokenizer):
    """After prefilling a prompt and rolling back to half, cache length halves."""
    model, tokenizer = target_model_and_tokenizer

    prompt = "The quick brown fox jumps over the lazy dog and then"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    cache = make_empty_cache()
    _, cache = prefill_cache(model, input_ids, cache)
    original_len = cache_seq_len(cache)

    rollback_to = original_len // 2
    rollback_cache(cache, rollback_to)

    assert cache_seq_len(cache) == rollback_to


def test_rollback_generation_matches_fresh_cache(target_model_and_tokenizer):
    """
    A cache polluted with a rejected branch, then rolled back, must behave
    exactly like a cache that never saw the branch.

    Path A: prefill the prompt, append five DELIBERATELY WRONG tokens (a
            distinctive unrelated phrase, standing in for draft proposals that
            the target rejects), roll back to the prompt length, then generate.
    Path B: prefill the prompt into a fresh cache, then generate.

    The two must agree token for token. Before comparing, the test asserts that
    the polluted branch really does change the model's next-token prediction —
    otherwise passing would mean nothing.
    """
    model, tokenizer = target_model_and_tokenizer

    prompt = "The capital city of France is"
    prefix_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    prefix_len = prefix_ids.shape[1]

    # A distinctive, semantically unrelated branch. These are the tokens that a
    # rejection would have to erase from the cache.
    wrong_text = " zebra quantum bicycle asparagus telescope"
    wrong_ids = tokenizer(wrong_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(DEVICE)
    n_wrong = wrong_ids.shape[1]
    assert n_wrong >= 3, "Need a branch of at least 3 tokens for this test to bite."

    # --- Path A: pollute, then roll back ---
    cache_a = make_empty_cache()
    logits_a, cache_a = prefill_cache(model, prefix_ids, cache_a)
    logit_at_prefix = logits_a[0, -1, :].clone()

    polluted_last_logit = logit_at_prefix
    for i in range(n_wrong):
        tok_t = wrong_ids[:, i : i + 1]
        new_logits, cache_a = single_token_forward(model, tok_t, cache_a)
        polluted_last_logit = new_logits[0, 0, :]

    assert cache_seq_len(cache_a) == prefix_len + n_wrong

    # Teeth check: the polluted context must actually change what comes next.
    # If it did not, a broken rollback would be indistinguishable from a
    # correct one and the assertion below would be vacuous.
    assert int(greedy_token(polluted_last_logit).item()) != int(
        greedy_token(logit_at_prefix).item()
    ), (
        "The polluting branch did not change the next-token prediction, so this "
        "test cannot detect a rollback failure. Choose a different branch."
    )

    rollback_cache(cache_a, prefix_len)
    assert cache_seq_len(cache_a) == prefix_len

    gen_from_rollback, _ = _greedy_continue(model, logit_at_prefix, cache_a, 12)

    # --- Path B: fresh cache, same prefix ---
    cache_b = make_empty_cache()
    logits_b, cache_b = prefill_cache(model, prefix_ids, cache_b)
    gen_from_fresh, _ = _greedy_continue(model, logits_b[0, -1, :], cache_b, 12)

    assert gen_from_rollback == gen_from_fresh, (
        "Rollback mismatch — the cache retained rejected tokens.\n"
        f"  from rolled-back cache: {gen_from_rollback}\n"
        f"  from fresh cache:       {gen_from_fresh}\n"
        "Speculative decoding would produce output conditioned on tokens it "
        "never emitted."
    )


def test_rollback_under_forced_rejections(target_model_and_tokenizer):
    """
    End-to-end: drive the real speculative decoder with a draft model whose
    proposals are almost always rejected, and require greedy equivalence.

    The draft is the genuine llama-68m but run against target prompts where it
    disagrees often; a run is only accepted as evidence if it actually produced
    rejections (mean acceptance length below K+1 on at least one round), which
    is asserted explicitly. Every rejection exercises a partial rollback to a
    length that is neither 0 nor the full block, which is where off-by-one
    errors live.
    """
    target_model, tokenizer = target_model_and_tokenizer
    draft_model, _ = load_draft_model(MODEL_DIR, device=DEVICE)

    K = 6
    prompt = (
        "In 1729 the mathematician Christian Goldbach wrote to Leonhard Euler "
        "about the conjecture that"
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    spec = generate_speculative(
        draft_model,
        target_model,
        input_ids.clone(),
        max_new_tokens=96,
        K=K,
        temperature=0.0,
        device=DEVICE,
    )
    base = generate_baseline(
        target_model,
        input_ids.clone(),
        max_new_tokens=96,
        temperature=0.0,
        device=DEVICE,
    )

    accepted = spec["accepted_per_round"]
    assert accepted, "No rounds were run."
    n_rejecting_rounds = sum(1 for a in accepted if a < K)
    assert n_rejecting_rounds > 0, (
        "No rejection occurred, so no partial rollback was exercised and this "
        "test proves nothing. Increase K or choose a harder prompt."
    )
    # Rejections at a range of positions is what makes this a real test of the
    # rollback arithmetic rather than of one fixed offset.
    assert len(set(accepted)) > 1, (
        f"All rounds accepted the same count ({accepted[0]}); only one rollback "
        "offset was exercised."
    )

    assert spec["generated_ids"] == base["generated_ids"], (
        f"Greedy speculative output diverged from baseline after "
        f"{n_rejecting_rounds} rejecting rounds — rollback is wrong.\n"
        f"  spec: {spec['generated_ids'][:20]}\n"
        f"  base: {base['generated_ids'][:20]}"
    )


def test_rollback_to_zero(target_model_and_tokenizer):
    """Rolling back to 0 empties the cache."""
    model, tokenizer = target_model_and_tokenizer
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"].to(DEVICE)

    cache = make_empty_cache()
    _, cache = prefill_cache(model, input_ids, cache)
    assert cache_seq_len(cache) == input_ids.shape[1]

    rollback_cache(cache, 0)
    assert cache_seq_len(cache) == 0


def test_partial_rollback_multiple_times(target_model_and_tokenizer):
    """Successive rollbacks to smaller lengths are stable."""
    model, tokenizer = target_model_and_tokenizer
    prompt = "Speculative decoding is a technique for accelerating language model inference."
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    cache = make_empty_cache()
    _, cache = prefill_cache(model, input_ids, cache)
    full_len = cache_seq_len(cache)

    for keep in [full_len, full_len - 2, full_len - 5, 3, 0]:
        rollback_cache(cache, keep)
        assert cache_seq_len(cache) == keep
