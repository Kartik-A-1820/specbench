"""
scripts/run_phase2_collect.py — Phase 2 step 1: cache target hidden states.

Runs the target model over the training corpus once and writes last-layer
features plus next-token ids to results/hidden_states/. The target is then not
needed again until the sweep, which is the point: it cannot share VRAM with the
head's optimiser state on a 3.4 GB card.

Skips itself if the manifest already reports the requested number of tokens, so
it is safe to re-run after an interruption.

Usage:
    python scripts/run_phase2_collect.py
    python scripts/run_phase2_collect.py --max-total-tokens 250000
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from specbench.eagle.collect import HIDDEN_STATES_DIR, collect_hidden_states
from specbench.models import load_target_model, verify_cuda_environment

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Collect target hidden states for EAGLE training.")
    parser.add_argument("--max-total-tokens", type=int, default=250_000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Re-collect even if a manifest exists.")
    parser.add_argument("--done-marker", type=str, default=None)
    args = parser.parse_args()

    manifest_path = HIDDEN_STATES_DIR / "manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        have = manifest.get("total_tokens", 0)
        if have >= args.max_total_tokens * 0.9:
            print(
                f"Hidden states already collected: {have:,} tokens across "
                f"{manifest.get('n_documents', 0)} documents. Skipping.",
                flush=True,
            )
            if manifest.get("degenerate_corpus"):
                print(
                    "WARNING: the existing cache was built from the degenerate "
                    "fallback corpus. Re-run with --force once network access to "
                    "the datasets is available, or do not report Phase 2.",
                    flush=True,
                )
            if args.done_marker:
                Path(args.done_marker).write_text("ok", encoding="utf-8")
            return
        print(f"Existing cache has only {have:,} tokens; re-collecting.", flush=True)

    verify_cuda_environment()

    print("Loading target model...", flush=True)
    target_model, tokenizer = load_target_model(MODEL_DIR, device="cuda")

    print("Collecting hidden states...", flush=True)
    result = collect_hidden_states(
        target_model,
        tokenizer,
        device="cuda",
        max_seq_len=args.max_seq_len,
        max_total_tokens=args.max_total_tokens,
    )
    print(
        f"Collected {result['total_tokens']:,} token positions from "
        f"{result['n_documents']} documents -> {result['output_dir']}",
        flush=True,
    )

    del target_model
    torch.cuda.empty_cache()

    if result["degenerate_corpus"]:
        print(
            "\nFAILED: no real corpus could be loaded, so the cached hidden "
            "states come from a handful of repeated sentences. Training a head "
            "on this would produce an acceptance rate that means nothing. "
            "Fix network access to the Hugging Face datasets and re-run.",
            flush=True,
        )
        sys.exit(2)

    if args.done_marker:
        Path(args.done_marker).write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
