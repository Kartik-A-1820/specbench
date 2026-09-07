"""
specbench/models.py — model loading, dtype/device policy, and VRAM probes.

Hardware context: GTX 1650 Ti Mobile (TU117), SM 7.5, 3.4 GB usable VRAM.
  - No FlashAttention-2 (requires SM 8.0+). Use SDPA.
  - No bfloat16. FP16 only.
  - No FP8, no tensor-core-dependent kernels.

Every entry point that uses this module should call verify_cuda_environment()
on startup so that a CPU-only torch build or wrong device is caught immediately.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Hardware assertions
# ---------------------------------------------------------------------------

EXPECTED_CAPABILITY = (7, 5)  # TU117 / Turing, the GTX 1650 Ti Mobile.


def verify_cuda_environment(strict_device: bool = False) -> None:
    """
    Check that the runtime environment can actually run the benchmark.

    Hard failure (AssertionError) if CUDA is unavailable. This is the single
    most common silent killer on Windows: `pip install torch` from PyPI
    resolves to a CPU-only wheel, every script then runs, and the numbers are
    quietly meaningless. Fail loudly and early.

    Device capability mismatch is a warning, not a failure. The implementation
    is correct on any CUDA device; only the tuning and the VRAM budget are
    calibrated for SM 7.5. Every run record carries the real device name and
    capability, so results measured elsewhere are still identifiable. Pass
    strict_device=True to make a mismatch fatal.

    Note on bf16: torch reports bf16 as "supported" on Turing via emulation, so
    it is not a usable discriminator for the device and is not checked here.
    """
    assert torch.cuda.is_available(), (
        "CUDA is not available to torch. The most likely cause is a CPU-only "
        "torch wheel (plain `pip install torch` on Windows resolves to one). "
        "Reinstall from the CUDA index:\n"
        "  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121\n"
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
    )
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_properties(0).name
    if tuple(cap) != EXPECTED_CAPABILITY:
        message = (
            f"Device is '{name}' with SM {cap[0]}.{cap[1]}; this project is "
            f"calibrated for SM {EXPECTED_CAPABILITY[0]}.{EXPECTED_CAPABILITY[1]} "
            "(GTX 1650 Ti Mobile). Results remain valid but the VRAM budget and "
            "chosen model pair were sized for that device."
        )
        if strict_device:
            raise AssertionError(message)
        print(f"WARNING: {message}", flush=True)
    print(
        f"CUDA OK: {name} (SM {cap[0]}.{cap[1]}), torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}",
        flush=True,
    )


def cuda_env_info() -> dict:
    """
    Return a provenance dict suitable for embedding in results files.
    Includes torch version, CUDA version, driver version, and device name.
    """
    driver_version = "unknown"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        driver_version = out.split("\n")[0].strip()
    except Exception:
        pass

    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": driver_version,
        "device": device.name if device else "cpu",
        "device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "device_total_memory_bytes": device.total_memory if device else 0,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


def load_target_model(
    model_dir: Optional[Path] = None,
    device: str = "cuda",
) -> tuple:
    """
    Load the target model (TinyLlama-1.1B-Chat-v1.0) in FP16 with SDPA attention.

    Returns (model, tokenizer). The model is moved to `device` after loading.
    Loading with device_map=None and then explicit .to(device) avoids accelerate
    dispatching weights to CPU, which would invalidate the VRAM measurements.
    """
    base = model_dir or DEFAULT_MODEL_DIR
    path = base / "TinyLlama-1.1B-Chat-v1.0"
    return _load_model(str(path), device=device)


def load_draft_model(
    model_dir: Optional[Path] = None,
    device: str = "cuda",
) -> tuple:
    """
    Load the draft model (llama-68m) in FP16 with SDPA attention.

    Returns (model, tokenizer).
    """
    base = model_dir or DEFAULT_MODEL_DIR
    path = base / "llama-68m"
    return _load_model(str(path), device=device)


def _load_model(path: str, device: str = "cuda") -> tuple:
    """
    Internal helper. Loads model and tokenizer from a local directory.

    Key choices:
      - torch_dtype=torch.float16: matches the target hardware (no bf16).
      - attn_implementation="sdpa": works on SM 7.5; flash_attention_2 requires SM 8.0+.
      - device_map=None + explicit .to(device): keeps the full model on one device
        so that torch.cuda.memory_allocated() gives an accurate weight footprint.
    """
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=None,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# VRAM probes
# ---------------------------------------------------------------------------

def vram_report(label: str = "") -> dict:
    """
    Return a snapshot of GPU memory usage.

    Fields:
      allocated_bytes:     currently allocated by torch tensors.
      max_allocated_bytes: peak allocation since last reset or process start.
      free_bytes:          free bytes as reported by the driver.
      total_bytes:         total device memory as reported by the driver.
    """
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    free, total = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated_bytes": allocated,
        "max_allocated_bytes": max_allocated,
        "free_bytes": free,
        "total_bytes": total,
        "used_bytes": total - free,
    }


def reset_peak_memory() -> None:
    """Reset the peak memory tracker. Call before a measurement interval."""
    torch.cuda.reset_peak_memory_stats()


def get_gpu_stats() -> dict:
    """
    Query nvidia-smi for current GPU temperature and clock frequency.
    Returns empty strings if nvidia-smi is unavailable.
    """
    result = {"gpu_temp_c": None, "gpu_clock_mhz": None}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        parts = out.split(",")
        if len(parts) >= 2:
            result["gpu_temp_c"] = int(parts[0].strip())
            result["gpu_clock_mhz"] = int(parts[1].strip())
    except Exception:
        pass
    return result
