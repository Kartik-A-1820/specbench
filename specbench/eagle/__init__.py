"""
specbench/eagle/__init__.py — EAGLE-style draft head (Phase 2).

EAGLE (Li et al., 2024) replaces the separate draft model with a small
head that autoregresses on the target model's hidden features rather than
on token ids. Because the head predicts in the target's representation
space, acceptance rates are substantially higher than a separate small model.

The accept/reject rule from Phase 1 (spec_decode.py) is unchanged. Only the
source of the draft distribution q changes. The distribution guarantee holds
identically.

Memory constraint forces an offline collection strategy:
  - collect.py runs the target over a training corpus and writes last-layer
    hidden states to disk (FP16, ~4 KB/token).
  - train.py loads the cached states and trains the head with the target
    model unloaded, fitting within the 3.4 GB VRAM budget.
  - spec_decode_eagle.py implements speculative decoding driven by the head.
"""
