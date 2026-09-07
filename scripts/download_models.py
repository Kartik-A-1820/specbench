"""
scripts/download_models.py — download both models to local disk.

Downloads TinyLlama-1.1B-Chat-v1.0 (target) and llama-68m (draft) to the
local models/ directory using huggingface_hub.snapshot_download.

After running this script, benchmarking works offline with HF_HUB_OFFLINE=1.
Safetensors format is preferred over pickle-based .bin files.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir /path/to/models
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Default: models/ at the project root.
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"

MODELS = [
    {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "local_name": "TinyLlama-1.1B-Chat-v1.0",
        "role": "target",
    },
    {
        "repo_id": "JackFram/llama-68m",
        "local_name": "llama-68m",
        "role": "draft",
    },
]


def download_model(repo_id: str, local_dir: Path) -> None:
    """Download a model from the Hub to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "flax_model.*", "tf_model.*", "rust_model.*"],
    )
    print(f"  Done: {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download specbench models.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to store models (default: {DEFAULT_MODEL_DIR}).",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model directory: {model_dir.resolve()}")

    for m in MODELS:
        local_dir = model_dir / m["local_name"]
        # "the directory is non-empty" is not good enough: an interrupted
        # download leaves a directory full of blobs with no usable weights, and
        # skipping it means the next stage fails with a confusing load error.
        # Require a config plus at least one weights file, and otherwise call
        # snapshot_download again — it resumes rather than restarting.
        has_config = (local_dir / "config.json").exists()
        has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
        if has_config and has_weights:
            print(f"\n{m['role'].title()} model already present at {local_dir}. Skipping.")
        else:
            if local_dir.exists():
                print(f"\n{local_dir} exists but is incomplete; resuming download.")
            download_model(m["repo_id"], local_dir)

    print("\nAll models downloaded. Set HF_HUB_OFFLINE=1 to confirm offline operation:")
    print("  $env:HF_HUB_OFFLINE=1  (PowerShell)")
    print("  python scripts/run_vram_probe.py")


if __name__ == "__main__":
    main()
