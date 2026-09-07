"""
specbench/eagle/collect.py — offline hidden-state collection for EAGLE training.

The target model cannot stay in VRAM alongside the draft head's training state
(weights + gradients + AdamW moments) on a 3.4 GB card. So the target runs over
the training corpus once, last-layer hidden states and token ids are written to
disk in FP16, the target is freed, and the head then trains against the cached
tensors. This is a hardware-forced design decision, documented as such in
docs/architecture.md.

Cache format, one file per document under results/hidden_states/:
    {
        "hidden_states": float16 tensor, shape (seq_len - 1, hidden_size),
        "token_ids":     int32 tensor,   shape (seq_len - 1,),
        "doc_id":        int,
    }

hidden_states[i] is the target's feature at absolute position i, and
token_ids[i] is the token at position i+1 — the thing that feature predicts.
eagle/train.py relies on exactly this alignment.

Cache size: hidden_dim 2048 x 2 bytes = 4 KB per token, so 250 K tokens is
about 1 GB on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

HIDDEN_STATES_DIR = Path(__file__).parent.parent.parent / "results" / "hidden_states"

# Corpus sources, tried in order. Prose and code are both included on purpose:
# half the benchmark prompt set is code, and a head trained on prose alone
# drafts code badly, which shows up directly as a lower acceptance rate on
# those prompts. Each entry is (dataset, config, split, text_field, n_docs).
CORPUS_SOURCES = [
    ("wikitext", "wikitext-2-raw-v1", "train", "text", 6000),
    ("mbpp", "full", "train", "code", 400),
]

# Fallback only. A handful of sentences is not a training corpus: a head fitted
# to it memorises them and drafts everything else badly, so any acceptance rate
# measured against it says nothing about the method. If collection falls back
# here the run is flagged degenerate, loudly, and Phase 2 must not be reported.
FALLBACK_CORPUS: List[str] = [
    "The study of machine learning encompasses a wide range of algorithms and techniques.",
    "Neural networks are computational systems loosely modeled on the human brain.",
    "Gradient descent is an optimization algorithm used to minimize the loss function.",
    "Transformers use attention mechanisms to process sequences in parallel.",
    "The backpropagation algorithm computes gradients efficiently using the chain rule.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n"
    "    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    "class Stack:\n    def __init__(self):\n        self.items = []\n"
    "    def push(self, item):\n        self.items.append(item)\n",
]


def build_corpus(min_chars: int = 200) -> tuple[List[str], dict]:
    """
    Assemble the EAGLE training corpus from public datasets.

    Returns (documents, provenance). `provenance` records which sources loaded
    and how many documents each contributed, so the training data behind any
    reported acceptance rate is reconstructable.

    A source that fails (no network, dataset moved, gated repo) is skipped with
    a warning rather than aborting the run. If every source fails, the
    degenerate fallback is returned and provenance["degenerate"] is True.
    """
    documents: List[str] = []
    provenance: dict = {"sources": [], "degenerate": False}

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "WARNING: the `datasets` package is not installed, so no real corpus "
            "can be loaded. Install it with: pip install datasets",
            flush=True,
        )
        return list(FALLBACK_CORPUS), {"sources": [], "degenerate": True}

    for name, config, split, field, n_docs in CORPUS_SOURCES:
        try:
            ds = load_dataset(name, config, split=split)
            texts = [t for t in ds[field] if isinstance(t, str) and len(t) >= min_chars]
            texts = texts[:n_docs]
            if not texts:
                raise ValueError(f"no documents of at least {min_chars} characters")
            documents.extend(texts)
            provenance["sources"].append(
                {"dataset": name, "config": config, "split": split, "documents": len(texts)}
            )
            print(f"  corpus: {name}/{config} -> {len(texts)} documents", flush=True)
        except Exception as exc:  # noqa: BLE001 — any source failure is non-fatal
            print(f"  WARNING: corpus source {name}/{config} unavailable: {exc}", flush=True)

    if not documents:
        print(
            "WARNING: no corpus source loaded; falling back to a handful of "
            "hand-written sentences. A head trained on these is meaningless and "
            "Phase 2 results must NOT be reported.",
            flush=True,
        )
        return list(FALLBACK_CORPUS), {"sources": [], "degenerate": True}

    return documents, provenance


def collect_hidden_states(
    target_model: torch.nn.Module,
    tokenizer,
    corpus: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    max_seq_len: int = 512,
    device: str = "cuda",
    max_total_tokens: Optional[int] = 250_000,
) -> dict:
    """
    Run the target model over a corpus, saving last-layer hidden states to disk.

    The target model must already be on `device`. After collection the caller
    should delete it and call torch.cuda.empty_cache() before training.

    Args:
        target_model:     The frozen target model, on `device`.
        tokenizer:        Tokenizer matching the target model.
        corpus:           Documents. Defaults to build_corpus().
        output_dir:       Where to write .pt files. Defaults to results/hidden_states/.
        max_seq_len:      Tokens per document; longer documents are truncated.
        device:           Device the target model is on.
        max_total_tokens: Stop once this many positions have been cached. Caps
                          both disk use and the length of an epoch; None means
                          use the whole corpus.

    Returns:
        dict with total_tokens, n_documents, degenerate_corpus, output_dir.
    """
    provenance: dict = {"sources": [], "degenerate": False}
    if corpus is None:
        corpus, provenance = build_corpus()
    if output_dir is None:
        output_dir = HIDDEN_STATES_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_model.eval()
    total_tokens = 0
    manifest = []

    for doc_id, text in enumerate(tqdm(corpus, desc="Collecting hidden states")):
        if max_total_tokens is not None and total_tokens >= max_total_tokens:
            break

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue  # Need at least one (feature, next-token) pair.

        with torch.no_grad():
            out = target_model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        # hidden[i] is the feature at position i; tokens[i] is the token at
        # position i+1, i.e. what that feature predicts. train.py depends on
        # this alignment — see the module docstring there.
        hidden = out.hidden_states[-1][0, :-1, :].half().cpu()
        tokens = input_ids[0, 1:].int().cpu()

        fname = output_dir / f"doc_{doc_id:06d}.pt"
        torch.save({"hidden_states": hidden, "token_ids": tokens, "doc_id": doc_id}, fname)

        total_tokens += int(hidden.shape[0])
        manifest.append({"doc_id": doc_id, "tokens": int(hidden.shape[0]), "file": fname.name})

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_tokens": total_tokens,
                "n_documents": len(manifest),
                "corpus_provenance": provenance,
                "degenerate_corpus": provenance.get("degenerate", False),
                "max_seq_len": max_seq_len,
                "documents": manifest,
            },
            f,
            indent=2,
        )

    return {
        "total_tokens": total_tokens,
        "n_documents": len(manifest),
        "degenerate_corpus": provenance.get("degenerate", False),
        "output_dir": str(output_dir),
    }


def load_hidden_state_dataset(data_dir: Optional[Path] = None) -> list:
    """
    Load all collected hidden-state files from disk.

    Returns a list of dicts, each with 'hidden_states' (float16) and
    'token_ids' (int32).
    """
    if data_dir is None:
        data_dir = HIDDEN_STATES_DIR
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("doc_*.pt"))
    dataset = []
    for f in files:
        dataset.append(torch.load(f, map_location="cpu", weights_only=True))
    return dataset
