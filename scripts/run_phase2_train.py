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

from specbench.eagle.head import ARCH_VERSION, EagleDraftHead
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
# The target model's own cross-entropy on the collected corpus, measured
# directly as mean(-log softmax(lm_head(cached_hidden))[next_token]), is about
# 2.98 (perplexity ~20). The head is trained to imitate that distribution, so
# anything much below it is not learning, it is leakage. A generous floor of
# half the teacher's CE flags the pathology without tripping on a head that is
# genuinely doing well.
MIN_PLAUSIBLE_CE = 1.5

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

    manifest_path = RESULTS_DIR / "eagle_checkpoints.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # A head is only skipped if its checkpoint exists AND was produced by the
    # current head architecture. Skipping on file existence alone means a fix to
    # the head silently reuses a checkpoint trained against the broken version.
    pending = []
    for c in selected:
        name = c["name"]
        have_file = (CHECKPOINTS_DIR / f"{name}_final.pt").exists()
        recorded = manifest.get(name, {}).get("arch_version")
        if have_file and recorded == ARCH_VERSION:
            print(f"Checkpoint for {name} exists at arch v{ARCH_VERSION}. Skipping.", flush=True)
            continue
        if have_file:
            print(
                f"Checkpoint for {name} was built with arch v{recorded}, current is "
                f"v{ARCH_VERSION}. Retraining.",
                flush=True,
            )
        pending.append(c)

    if not pending:
        print("All requested heads already trained at the current architecture.", flush=True)
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

        # Sanity floor on the cross-entropy. The head is trained to imitate the
        # target model's next-token distribution, so it cannot legitimately beat
        # the target's own cross-entropy on the same data — a student does not
        # out-predict its teacher at reproducing the teacher. A CE far below
        # that floor means the head is reading the label from somewhere it
        # should not, which is what a missing causal mask does. Loud, because
        # the alternative symptom is a Phase 2 benchmark that quietly shows no
        # speedup and gets misread as "EAGLE does not help here".
        final_ce = result["history"]["ce_loss"][-1]
        if final_ce < MIN_PLAUSIBLE_CE:
            print(
                f"\n  WARNING: final cross-entropy {final_ce:.4f} is below the "
                f"plausibility floor of {MIN_PLAUSIBLE_CE}. The head cannot predict "
                "the target's own next token better than the target does. This "
                "almost always means label leakage — check that attention inside "
                "the head is causal (tests/test_eagle_head.py) before trusting any "
                "Phase 2 result.\n",
                flush=True,
            )

        ckpt = CHECKPOINTS_DIR / f"{cfg['name']}_final.pt"
        torch.save(head.state_dict(), ckpt)
        manifest[cfg["name"]] = {
            "checkpoint": str(ckpt),
            "arch_version": ARCH_VERSION,
            "ce_below_plausibility_floor": bool(final_ce < MIN_PLAUSIBLE_CE),
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
