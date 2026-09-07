"""
specbench/eagle/head.py — EAGLE-style draft head module.

The draft head takes the target model's last-layer hidden state and the
current token embedding and predicts the next hidden state. The target's
frozen LM head converts that predicted hidden state to logits.

Architecture (matching TinyLlama-1.1B-Chat-v1.0 dimensions):
  - hidden_dim = 2048 (target's hidden size)
  - intermediate_dim = 5632 (target's intermediate size, configurable)
  - Input: concat(hidden_state, token_embedding) → shape (batch, 4096)
  - fc_in: Linear(4096 → 2048)
  - decoder_layer: one (or more) LlamaDecoderLayer at hidden 2048
  - Output: predicted next hidden state → shape (batch, 2048)

The target's token embedding matrix (shape vocab × hidden_dim) and LM head
(Linear(hidden_dim, vocab)) are frozen and shared — not copied into this module.

Trainable parameters: fc_in + decoder layer(s) (~60 M for 1 layer).

Loss:
  L = α * SmoothL1(predicted_hidden, target_hidden) + (1-α) * CrossEntropy(logits, next_token)

where α is a hyperparameter (default 0.1). Both loss terms are reported
separately during training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
)


# Bumped whenever a change to this module invalidates existing checkpoints.
# scripts/run_phase2_train.py records it in the checkpoint manifest and retrains
# any head whose recorded version differs, so a fix to the architecture cannot
# be silently skipped by the "checkpoint already exists" resumability check.
#
#   1 — initial version.
#   2 — explicit causal mask in forward(). Version 1 heads were trained with
#       bidirectional attention and are worthless; they must be retrained.
ARCH_VERSION = 2


class EagleDraftHead(nn.Module):
    """
    EAGLE-style autoregressive draft head.

    Args:
        config:            LlamaConfig from the target model (provides hidden_size,
                           intermediate_size, num_attention_heads, etc.).
        num_draft_layers:  Number of LlamaDecoderLayer blocks in the head (default 1).
        intermediate_size: Override the intermediate MLP size (default: from config).
    """

    def __init__(
        self,
        config: LlamaConfig,
        num_draft_layers: int = 1,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_draft_layers = num_draft_layers

        # Input projection: concat(hidden_state, token_emb) → hidden_size.
        self.fc_in = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        # Build a modified config for the draft layers.
        draft_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size or config.intermediate_size,
            num_hidden_layers=num_draft_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            attention_bias=getattr(config, "attention_bias", False),
            mlp_bias=getattr(config, "mlp_bias", False),
        )
        self.draft_config = draft_config

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(draft_config, layer_idx=i)
            for i in range(num_draft_layers)
        ])
        self.norm = LlamaRMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))

        # Rotary embedding, required by LlamaDecoderLayer in transformers >= 4.46.
        #
        # The config must be passed by KEYWORD. LlamaRotaryEmbedding's first
        # positional parameter is still `dim` for backwards compatibility, so
        # passing the config positionally silently binds it to `dim` and blows
        # up inside torch.arange with a confusing type error.
        self.rotary_emb = LlamaRotaryEmbedding(config=draft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Predict the next hidden state.

        Args:
            hidden_states: Target's last-layer hidden states, shape (batch, seq, hidden_size).
            token_embeds:  Token embeddings for the current tokens, shape (batch, seq, hidden_size).
            attention_mask: Optional additive attention mask. If None, a causal
                            mask is built here — see the note below, this must
                            not be left to the layer.
            position_ids:  Optional position ids. Required when `past_key_values`
                           is supplied, because the new positions no longer start
                           at zero.
            past_key_values: Optional Cache. The head is an autoregressive
                           decoder, so at inference it must carry a KV cache
                           across draft steps exactly as the target does.
                           Without one, the head is called on a length-1
                           sequence with no history — a computation it never
                           performs during training, where it sees whole chunks
                           and can attend over them. Measured cost of getting
                           this wrong on this project: acceptance falls from
                           0.279 to 0.231.
            use_cache:     Whether to write new keys/values into past_key_values.

        Returns:
            predicted_hidden: shape (batch, seq, hidden_size).
        """
        # Concatenate hidden state and token embedding along feature dimension.
        x = torch.cat([hidden_states, token_embeds], dim=-1)  # (batch, seq, 2*hidden)
        x = self.fc_in(x)  # (batch, seq, hidden)

        batch, seq_len, _ = x.shape
        device = x.device

        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()

        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=device)
                .unsqueeze(0)
                .expand(batch, -1)
            )

        # Compute rotary position embeddings required by LlamaDecoderLayer.
        position_embeddings = self.rotary_emb(x, position_ids)

        # Build an EXPLICIT causal mask when the caller did not supply one.
        #
        # This is not defensive tidying; leaving it out is a silent correctness
        # disaster during training. A bare LlamaDecoderLayer built from a plain
        # LlamaConfig uses the "eager" attention path, and eager attention
        # applies no mask at all when attention_mask is None — attention becomes
        # bidirectional. Training then leaks the answer: the label for position
        # t is the input token at position t+1, so the head learns to read it
        # off the future instead of predicting it. Cross-entropy collapses to
        # near zero, training looks excellent, and at inference — where the head
        # runs one position at a time and has no future to read — the head is
        # useless and acceptance goes to nothing.
        #
        # Constructing the mask here rather than relying on `is_causal` makes
        # the behaviour independent of which attention implementation the config
        # happens to select.
        if attention_mask is None and seq_len > 1:
            total_len = past_len + seq_len
            causal = torch.full(
                (seq_len, total_len), torch.finfo(x.dtype).min, device=device, dtype=x.dtype
            ).triu(diagonal=1 + past_len)
            attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, total_len)

        cache_position = torch.arange(past_len, past_len + seq_len, device=device)

        # Run through decoder layers.
        for layer in self.layers:
            layer_out = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            x = layer_out[0]

        x = self.norm(x)
        return x  # predicted hidden state

    @staticmethod
    def compute_loss(
        predicted_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        logits: torch.Tensor,
        next_token_ids: torch.Tensor,
        alpha: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss.

        Args:
            predicted_hidden: Head's output, shape (batch, seq, hidden).
            target_hidden:    Ground-truth next hidden state, shape (batch, seq, hidden).
            logits:           LM head output from predicted_hidden, shape (batch, seq, vocab).
            next_token_ids:   Ground-truth next token ids, shape (batch, seq).
            alpha:            Weight for the regression term (default 0.1).

        Returns:
            (total_loss, regression_loss, ce_loss)
        """
        # Regression loss on hidden state.
        reg_loss = nn.functional.smooth_l1_loss(predicted_hidden, target_hidden.float())

        # Cross-entropy loss on token prediction.
        batch, seq, vocab = logits.shape
        ce_loss = nn.functional.cross_entropy(
            logits.view(batch * seq, vocab).float(),
            next_token_ids.view(batch * seq),
            ignore_index=-100,
        )

        total_loss = alpha * reg_loss + (1.0 - alpha) * ce_loss
        return total_loss, reg_loss, ce_loss


def build_eagle_head(
    target_model: nn.Module,
    num_draft_layers: int = 1,
    intermediate_size: Optional[int] = None,
) -> EagleDraftHead:
    """
    Construct an EagleDraftHead sized to match the target model.

    Args:
        target_model:      The target LlamaForCausalLM (used only for its config).
        num_draft_layers:  Number of decoder layers in the head.
        intermediate_size: Override intermediate MLP size.
    """
    config = target_model.config
    return EagleDraftHead(
        config=config,
        num_draft_layers=num_draft_layers,
        intermediate_size=intermediate_size,
    )
