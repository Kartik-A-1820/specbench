"""
scripts/run_phase2_train.py — Phase 2 step 2: train EAGLE draft heads.

Trains one head per capacity configuration against the cached hidden states.
The target model body is never loaded here: only its token embedding and LM
head are needed, both frozen, both small. That is what keeps head training
inside the VRAM budget.

Each configuration writes results/checkpoints/<name>_final.pt and is skipped if
that file already exists, so an interrupted run resumes at the next untrained
configuration rather than starting over.

Usage:
    python scripts/run_phase2_train.py
    python scripts/run_phase2_train.py --configs eagle_L1_I5632 --epochs 3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from specbench.eagle.head import EagleDraftHead
from specbench.eagle.train import train_eagle_head
from specbench.models import load_target_model, verify_cuda_environment

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Capacity sweep. The pairs here are the ones that fit: one and two decoder
# layers, at the target's own intermediate width and at half of it. Anything
# larger stops fitting alongside AdamW state on 3.4 GB, which is itself a
# finding and is recorded in docs/vram-budget.md.
HEAD_CONFIGS = [
    {"name": "eagle_L1_I2816", "num_layers": 1, "intermediate_size": 2816},
    {"name": "eagle_L1_I5632", "num_layers": 1, "intermediate_size": 5632},
    {"name": "eagle_L2_I2816", "num_layers": 2, "intermediate_size": 2816},
    {"name": "eagle_L2_I5632", "num_layers": 2, "intermediate_size": 5632},
]


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE draft heads.")
    parser.add_argument("--configs", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--done-marker", type=str, default=None)
    args = parser.parse_args()

    selected = HEAD_CONFIGS
    if args.configs:
        selected = [c for c in HEAD_CONFIGS if c["name"] in args.configs]
        if not selected:
            print(f"No configs matched {args.configs}.", flush=True)
            sys.exit(1)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    pending = [c for c in selected if not (CHECKPOINTS_DIR / f"{c['name']}_final.pt").exists()]
    for c in selected:
        if c not in pending:
            print(f"Checkpoint for {c['name']} already exists. Skipping.", flush=True)
    if not pending:
        print("All requested heads already trained.", flush=True)
        if args.done_marker:
            Path(args.done_marker).write_text("ok", encoding="utf-8")
        return

    verify_cuda_environment()

    # Load the target only to lift out its config and frozen components, then
    # drop the body before any optimiser state is allocated.
    print("Loading target model to extract config, embeddings and LM head...", flush=True)
    target_model, _ = load_target_model(MODEL_DIR, device="cpu")
    target_config = target_model.config
    embed_state = {k: v.clone() for k, v in target_model.model.embed_tokens.state_dict().items()}
    lm_head_weight_cpu = target_model.lm_head.weight.detach().clone()
    del target_model
    torch.cuda.empty_cache()
    print("Target body released.", flush=True)

    manifest_path = RESULTS_DIR / "eagle_checkpoints.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for cfg in pending:
        print(f"\n{'=' * 60}\nTraining {cfg['name']}\n{'=' * 60}", flush=True)

        emb = nn.Embedding(target_config.vocab_size, target_config.hidden_size)
        emb.load_state_dict(embed_state)
        emb = emb.to("cuda").half()
        emb.requires_grad_(False)

        head = EagleDraftHead(
            config=target_config,
            num_draft_layers=cfg["num_layers"],
            intermediate_size=cfg["intermediate_size"],
        )
        n_params = sum(p.numel() for p in head.parameters())
        print(f"  head parameters: {n_params / 1e6:.1f} M", flush=True)

        result = train_eagle_head(
            head=head,
            token_embedding=emb,
            lm_head=lm_head_weight_cpu,
            epochs=args.epochs,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            lr=args.lr,
            alpha=args.alpha,
            device="cuda",
            run_name=cfg["name"],
        )
        print(f"  final total loss: {result['history']['total_loss'][-1]:.4f}", flush=True)

        ckpt = CHECKPOINTS_DIR / f"{cfg['name']}_final.pt"
        torch.save(head.state_dict(), ckpt)
        manifest[cfg["name"]] = {
            "checkpoint": str(ckpt),
            "num_layers": cfg["num_layers"],
            "intermediate_size": cfg["intermediate_size"],
            "parameters": int(n_params),
            "final_total_loss": result["history"]["total_loss"][-1],
            "final_ce_loss": result["history"]["ce_loss"][-1],
            "final_reg_loss": result["history"]["reg_loss"][-1],
            "peak_vram_bytes": result["peak_vram_bytes"],
            "epochs": args.epochs,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  saved {ckpt}", flush=True)

        del head, emb
        torch.cuda.empty_cache()

    if args.done_marker:
        Path(args.done_marker).write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
