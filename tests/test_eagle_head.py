"""
tests/test_eagle_head.py — EAGLE head structural correctness.

These tests run on CPU with a randomly initialised head. They are not about
whether the head predicts well; they are about whether it is being trained on
the task it will actually face at inference.

The bug that motivated this file: a bare LlamaDecoderLayer built from a plain
LlamaConfig uses the "eager" attention path, and eager attention applies no
mask at all when attention_mask is None. Attention was therefore bidirectional
during training. Since the label for position t is the input token at position
t+1, the head could read its answer off the future. Cross-entropy collapsed to
0.05 — against a floor of about 2.98, which is the target model's own
cross-entropy on the same data, and no student can beat its teacher at
imitating the teacher. At inference the head sees one position at a time with
no future to read, so the learned strategy is worthless and acceptance
collapses.

Nothing crashes when this happens. Training loss looks superb. The only signals
are a loss below the teacher's floor and a benchmark that shows no speedup —
which is easy to misread as "EAGLE does not help on this hardware".
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from specbench.eagle.head import EagleDraftHead


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture(scope="module")
def head() -> EagleDraftHead:
    torch.manual_seed(0)
    h = EagleDraftHead(_tiny_config(), num_draft_layers=1)
    h.eval()
    return h


def test_head_is_causal(head):
    """
    The output at position t must not depend on inputs at positions > t.

    Perturb the last position's inputs and require every earlier output to be
    bit-identical. If attention is bidirectional, earlier positions move and
    this fails.
    """
    torch.manual_seed(1)
    B, T, H = 1, 12, head.hidden_size
    hidden = torch.randn(B, T, H)
    embeds = torch.randn(B, T, H)

    with torch.no_grad():
        base = head(hidden, embeds)

    hidden_perturbed = hidden.clone()
    embeds_perturbed = embeds.clone()
    hidden_perturbed[:, -1, :] += 100.0
    embeds_perturbed[:, -1, :] += 100.0

    with torch.no_grad():
        perturbed = head(hidden_perturbed, embeds_perturbed)

    max_drift = (base[:, :-1, :] - perturbed[:, :-1, :]).abs().max().item()
    assert max_drift == 0.0, (
        f"Changing the input at the LAST position changed the output at earlier "
        f"positions by up to {max_drift:.3e}. Attention is not causal, so during "
        "training the head can read the label for position t from the input at "
        "position t+1. Training cross-entropy will collapse far below the target "
        "model's own cross-entropy, and the head will be useless at inference."
    )


def test_head_causality_at_every_position(head):
    """
    Stronger form: perturbing position j must leave every output at t < j
    untouched, for every j.
    """
    torch.manual_seed(2)
    B, T, H = 1, 8, head.hidden_size
    hidden = torch.randn(B, T, H)
    embeds = torch.randn(B, T, H)
    with torch.no_grad():
        base = head(hidden, embeds)

    for j in range(1, T):
        hp = hidden.clone()
        ep = embeds.clone()
        hp[:, j, :] += 50.0
        ep[:, j, :] += 50.0
        with torch.no_grad():
            out = head(hp, ep)
        drift = (base[:, :j, :] - out[:, :j, :]).abs().max().item()
        assert drift == 0.0, (
            f"Perturbing position {j} moved outputs at positions < {j} by "
            f"{drift:.3e}. Attention leaks from the future."
        )


def test_single_position_forward_matches_batched_prefix(head):
    """
    Running the head over a length-1 sequence (the inference path) must agree
    with position 0 of a batched forward (the training path).

    Training and inference use different sequence lengths, so a discrepancy
    here means the head is optimised for a computation it never performs.
    """
    torch.manual_seed(3)
    B, T, H = 1, 6, head.hidden_size
    hidden = torch.randn(B, T, H)
    embeds = torch.randn(B, T, H)

    with torch.no_grad():
        batched = head(hidden, embeds)
        single = head(hidden[:, :1, :], embeds[:, :1, :])

    diff = (batched[:, 0, :] - single[:, 0, :]).abs().max().item()
    assert diff < 1e-5, (
        f"Position 0 differs by {diff:.3e} between a batched forward and a "
        "length-1 forward. The training-time and inference-time computations "
        "have diverged."
    )


def test_output_shape(head):
    B, T, H = 2, 5, head.hidden_size
    with torch.no_grad():
        out = head(torch.randn(B, T, H), torch.randn(B, T, H))
    assert out.shape == (B, T, H)
