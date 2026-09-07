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

        # Rotary embedding (required by LlamaDecoderLayer in transformers>=4.46).
        self.rotary_emb = LlamaRotaryEmbedding(draft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict the next hidden state.

        Args:
            hidden_states: Target's last-layer hidden states, shape (batch, seq, hidden_size).
            token_embeds:  Token embeddings for the current tokens, shape (batch, seq, hidden_size).
            attention_mask: Optional attention mask.
            position_ids:  Optional position ids.

        Returns:
            predicted_hidden: shape (batch, seq, hidden_size).
        """
        # Concatenate hidden state and token embedding along feature dimension.
        x = torch.cat([hidden_states, token_embeds], dim=-1)  # (batch, seq, 2*hidden)
        x = self.fc_in(x)  # (batch, seq, hidden)

        batch, seq_len, _ = x.shape
        device = x.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        # Compute rotary position embeddings required by LlamaDecoderLayer.
        position_embeddings = self.rotary_emb(x, position_ids)

        # Run through decoder layers.
        for layer in self.layers:
            layer_out = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
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
