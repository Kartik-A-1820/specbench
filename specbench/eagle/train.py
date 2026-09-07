"""
specbench/eagle/train.py — EAGLE draft head training loop.

Training runs with the target model UNLOADED from VRAM. The target's weights
are needed only for two frozen components: the token embedding matrix and the
LM head. These are small enough to load separately (about 0.13 GB combined for
TinyLlama-1.1B) and kept frozen throughout training.

VRAM budget during training (without target model body):
    Draft head params (~60 M, FP32)     0.24 GB
    Gradients                           0.24 GB
    AdamW m, v states                   0.48 GB
    Token embeddings + LM head (FP16)   0.13 GB
    Activations (batch 4, seq 512)     ~0.20 GB
                                        -------
    Total                              ~1.29 GB   (well within 3.4 GB)

Loss function: alpha * SmoothL1(predicted_hidden, target_hidden)
             + (1-alpha) * CrossEntropy(logits, next_token_id)
where alpha defaults to 0.1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from specbench.eagle.collect import load_hidden_state_dataset
from specbench.eagle.head import EagleDraftHead
from specbench.models import reset_peak_memory, vram_report

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


class HiddenStateDataset(Dataset):
    """
    Dataset wrapping the offline-collected hidden state files.

    Each item is a (hidden_state, token_id, next_token_id) triple at the
    sequence level — i.e., one training sample per (position, document) pair.
    Items are grouped into fixed-length chunks for batching.
    """

    def __init__(self, data: list, chunk_size: int = 64) -> None:
        self.chunks = []
        for doc in data:
            hidden = doc["hidden_states"]  # (L, hidden)
            tokens = doc["token_ids"]       # (L,) — these are the NEXT tokens
            L = hidden.shape[0]
            if L < 2:
                continue
            # Split into chunks of size chunk_size (drop remainder).
            for start in range(0, L - chunk_size, chunk_size):
                end = start + chunk_size
                self.chunks.append({
                    "hidden": hidden[start:end],        # (chunk, hidden)
                    "next_tokens": tokens[start:end],   # (chunk,)
                    # The "current" hidden for input is the previous position.
                    # For position 0 within a chunk, use zero vector.
                    "prev_hidden": torch.cat([
                        torch.zeros(1, hidden.shape[1], dtype=hidden.dtype),
                        hidden[start:end - 1],
                    ], dim=0),                          # (chunk, hidden)
                    "current_tokens": torch.cat([
                        tokens[start:start + 1] - tokens[start:start + 1],  # zero placeholder
                        tokens[start:end - 1],
                    ], dim=0),                          # (chunk,) — the current (not next) token ids
                })

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        return self.chunks[idx]


def train_eagle_head(
    head: EagleDraftHead,
    token_embedding: nn.Embedding,
    lm_head: nn.Linear,
    data_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 3e-4,
    alpha: float = 0.1,
    chunk_size: int = 64,
    device: str = "cuda",
    checkpoint_every: int = 1,
    seed: int = 42,
    run_name: str = "eagle_head",
) -> dict:
    """
    Train the EAGLE draft head against offline-collected hidden states.

    Args:
        head:            The EagleDraftHead to train (will be moved to device).
        token_embedding: Frozen target token embedding (nn.Embedding).
        lm_head:         Frozen target LM head (nn.Linear).
        data_dir:        Directory with collected .pt files.
        epochs:          Training epochs.
        batch_size:      Batch size for DataLoader.
        lr:              AdamW learning rate.
        alpha:           Regression loss weight (see EagleDraftHead.compute_loss).
        chunk_size:      Sequence length for training chunks.
        device:          "cuda" or "cpu".
        checkpoint_every: Save a checkpoint every N epochs.
        seed:            Random seed.

    Returns:
        dict with training history and peak VRAM.
    """
    torch.manual_seed(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Move head and frozen components to device.
    head = head.to(device).float()  # Train in float32.
    token_embedding = token_embedding.to(device).half()   # Keep frozen in FP16.

    # lm_head may be an nn.Linear or a raw weight tensor.
    if isinstance(lm_head, nn.Linear):
        lm_head_weight = lm_head.weight.to(device).half()
    elif isinstance(lm_head, torch.Tensor):
        lm_head_weight = lm_head.to(device).half()
    else:
        raise TypeError(f"lm_head must be nn.Linear or torch.Tensor, got {type(lm_head)}")

    # Freeze embedding and LM head.
    token_embedding.requires_grad_(False)
    lm_head_weight.requires_grad_(False)

    # Load dataset.
    print("Loading hidden state dataset from disk...")
    raw_data = load_hidden_state_dataset(data_dir)
    if not raw_data:
        raise RuntimeError(
            f"No hidden state files found in {data_dir or 'results/hidden_states/'}. "
            "Run scripts/run_phase2_collect.py first."
        )
    dataset = HiddenStateDataset(raw_data, chunk_size=chunk_size)
    print(f"Dataset: {len(dataset)} chunks from {len(raw_data)} documents.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows: no multiprocessing for DataLoader.
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    reset_peak_memory()
    history = {"reg_loss": [], "ce_loss": [], "total_loss": [], "epoch_times": []}

    for epoch in range(1, epochs + 1):
        head.train()
        epoch_reg = []
        epoch_ce = []
        epoch_total = []
        t0 = time.perf_counter()

        for batch in loader:
            prev_hidden = batch["prev_hidden"].to(device).float()   # (B, seq, hidden)
            target_hidden = batch["hidden"].to(device).float()       # (B, seq, hidden)
            current_tokens = batch["current_tokens"].to(device)      # (B, seq)
            next_tokens = batch["next_tokens"].to(device).long()     # (B, seq)

            # Token embeddings for current tokens.
            with torch.no_grad():
                tok_emb = token_embedding(current_tokens).float()    # (B, seq, hidden)

            # Forward through head.
            predicted_hidden = head(prev_hidden, tok_emb)            # (B, seq, hidden)

            # LM head: use FP16 weight, cast predicted hidden to FP16 for matmul.
            logits = predicted_hidden.half() @ lm_head_weight.t()   # (B, seq, vocab)
            logits = logits.float()

            total_loss, reg_loss, ce_loss = EagleDraftHead.compute_loss(
                predicted_hidden, target_hidden, logits, next_tokens, alpha=alpha
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_reg.append(reg_loss.item())
            epoch_ce.append(ce_loss.item())
            epoch_total.append(total_loss.item())

        epoch_time = time.perf_counter() - t0
        history["reg_loss"].append(float(np.mean(epoch_reg)))
        history["ce_loss"].append(float(np.mean(epoch_ce)))
        history["total_loss"].append(float(np.mean(epoch_total)))
        history["epoch_times"].append(epoch_time)

        vr = vram_report(f"epoch_{epoch}")
        print(
            f"Epoch {epoch}/{epochs} | "
            f"total={history['total_loss'][-1]:.4f} "
            f"reg={history['reg_loss'][-1]:.4f} "
            f"ce={history['ce_loss'][-1]:.4f} "
            f"| {epoch_time:.1f}s | "
            f"VRAM {vr['max_allocated_bytes'] / 1e9:.2f} GB"
        )

        if epoch % checkpoint_every == 0:
            ckpt_path = CHECKPOINTS_DIR / f"{run_name}_epoch{epoch}.pt"
            torch.save(head.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    _save_loss_curves(history, run_name)

    peak_vram = vram_report("training_peak")
    result = {
        "run_name": run_name,
        "history": history,
        "peak_vram_bytes": peak_vram["max_allocated_bytes"],
        "epochs": epochs,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "lr": lr,
        "alpha": alpha,
        "n_chunks": len(dataset),
        "n_documents": len(raw_data),
    }

    # One summary file per run, merged into a shared index so that a sweep does
    # not overwrite earlier configurations' training records.
    index_path = RESULTS_DIR / "eagle_training_summary.json"
    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            index = {}
    index[run_name] = result
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return result


def _save_loss_curves(history: dict, run_name: str = "eagle_head") -> None:
    """Save the loss curve plot to results/figures/<run_name>_loss_curves.png."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # No display on a headless batch run.
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["total_loss"]) + 1)

        axes[0].plot(epochs, history["total_loss"], label="total", color="black")
        axes[0].plot(epochs, history["reg_loss"], label="regression (smooth L1)", linestyle="--")
        axes[0].plot(epochs, history["ce_loss"], label="cross-entropy", linestyle=":")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"EAGLE Head Training Loss — {run_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["epoch_times"], color="orange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Wall-clock time (s)")
        axes[1].set_title("Training Time per Epoch")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / f"{run_name}_loss_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curves saved to {out}")
    except Exception as e:  # noqa: BLE001 — plotting must never fail a training run
        print(f"Warning: could not save loss curves: {e}")
