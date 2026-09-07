"""
scripts/check_env.py — fail fast, and loudly, on a broken environment.

Run first by run_all.bat. The failure this exists to catch is a CPU-only torch
wheel: `pip install torch` on Windows resolves to one, every subsequent script
then runs to completion, and every number produced is meaningless. That must be
an immediate hard stop, not a warning buried in a log.

Exit codes:
    0  environment is usable
    1  CUDA unavailable, or a required package is missing
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REQUIRED = [
    "torch",
    "transformers",
    "numpy",
    "scipy",
    "matplotlib",
    "huggingface_hub",
    "tqdm",
    "pytest",
]
OPTIONAL = ["datasets"]  # Only Phase 2 needs this.


def main() -> int:
    print(f"platform: {platform.platform()}")
    print(f"python:   {sys.version.splitlines()[0]}")
    print(f"executable: {sys.executable}")

    missing = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:18s} {getattr(mod, '__version__', 'unknown')}")
        except ImportError as exc:
            print(f"  {name:18s} MISSING ({exc})")
            missing.append(name)

    for name in OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:18s} {getattr(mod, '__version__', 'unknown')} (optional)")
        except ImportError:
            print(
                f"  {name:18s} MISSING (optional; Phase 2 EAGLE training needs it: "
                f"pip install {name})"
            )

    if missing:
        print("\nFAILED: install the missing packages:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        return 1

    import torch

    print(f"\ntorch.__version__      = {torch.__version__}")
    print(f"torch.version.cuda     = {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    if torch.version.cuda is None:
        print(
            "\n"
            "================================================================\n"
            "FAILED: this is a CPU-ONLY BUILD OF TORCH.\n"
            "torch.version.cuda is None, so no CUDA kernels are compiled in.\n"
            "Every benchmark would run on the CPU and every number would be\n"
            "meaningless. Reinstall from the CUDA wheel index:\n"
            "\n"
            f"  {sys.executable} -m pip uninstall -y torch\n"
            f"  {sys.executable} -m pip install torch==2.5.1 "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "================================================================"
        )
        return 1

    if not torch.cuda.is_available():
        print(
            "\n"
            "================================================================\n"
            "FAILED: torch has CUDA support compiled in, but no CUDA device is\n"
            "visible. Common causes:\n"
            "  - the NVIDIA driver is older than the CUDA runtime in the wheel\n"
            "  - CUDA_VISIBLE_DEVICES is set to an empty value\n"
            "  - the GPU is in use by another process that has exhausted it\n"
            "Run `nvidia-smi` to confirm the device and driver are healthy.\n"
            "================================================================"
        )
        return 1

    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability()
    print(f"device:   {props.name}")
    print(f"capability: SM {cap[0]}.{cap[1]}")
    print(f"total VRAM: {props.total_memory / 1e9:.2f} GB")

    # A real allocation and matmul: is_available() can be true on a device that
    # then fails on first use.
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        b = (a @ a).float().sum().item()
        torch.cuda.synchronize()
        print(f"fp16 matmul smoke test: OK (checksum {b:.1f})")
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: a trivial fp16 matmul on the GPU raised: {exc}")
        return 1

    if tuple(cap) != (7, 5):
        print(
            f"\nNOTE: this project was calibrated for SM 7.5 (GTX 1650 Ti Mobile); "
            f"this device is SM {cap[0]}.{cap[1]}. Results stay valid — the device "
            "name and capability are recorded in every run record — but the VRAM "
            "budget and model pair were chosen for that card."
        )

    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
