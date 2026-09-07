"""
specbench/harness.py — timing, metric collection, and run records.

Every benchmark run appends one JSON object per generation to
results/raw_measurements.jsonl. Aggregates are merged into
results/summary.json, keyed by a tag identifying the configuration, so that a
later configuration never overwrites an earlier one.

The harness enforces an interleaved measurement design: for each prompt every
arm (baseline plus each speculative configuration) runs once, back to back,
and the order of arms is rotated from prompt to prompt. Two things follow.
Thermal drift and clock throttling on a laptop are slow relative to a single
prompt, so they hit all arms roughly equally instead of penalising whichever
condition happened to run last. And rotating the order stops the first arm in
the sequence from always paying the cold-cache cost. See docs/methodology.md.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from specbench.models import cuda_env_info, get_gpu_stats
from specbench.stats import (
    bootstrap_mean_ci,
    bootstrap_speedup_ci,
    latency_percentiles,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_JSONL = RESULTS_DIR / "raw_measurements.jsonl"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

# Minimum number of warmup runs to discard (per condition).
WARMUP_RUNS = 2


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _append_record(record: dict) -> None:
    """Append a single JSON record to raw_measurements.jsonl."""
    _ensure_results_dir()
    with open(RAW_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def merge_summary(tag: str, summary: dict) -> None:
    """
    Merge one configuration's summary into results/summary.json under `tag`.

    Read-modify-write rather than overwrite: a sweep runs many configurations
    and each must survive the next. It also means an interrupted sweep leaves
    every completed configuration intact on disk.
    """
    _ensure_results_dir()
    existing: Dict[str, Any] = {}
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except json.JSONDecodeError:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[tag] = summary
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Timed generation wrapper
# ---------------------------------------------------------------------------

def _synchronize() -> None:
    """Block until all queued CUDA work has completed. No-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def derive_latencies(result: dict) -> dict:
    """
    Turn the measured per-round timestamps into latency series.

    The generators record `round_end_times_s`: the instant, relative to the
    start of generation, at which each round finished. Speculative decoding
    emits every token of a round at that same instant — they all come out of
    one verification pass — so there are two distinct things worth measuring
    and they are reported separately rather than blended:

      token_arrival_gaps_ms
          Differences between consecutive token arrival times. Tokens after
          the first in a round have a gap of 0.0 because they genuinely became
          available simultaneously. This is what a streaming client sees.

      round_latencies_ms
          Wall-clock duration of each round. For the baseline a round is one
          token, so this reduces to conventional inter-token latency; for the
          speculative path it is the burst period.

    Nothing here is inferred from a mean. If `round_end_times_s` is absent or
    empty (generator run without a clock) both series come back empty rather
    than being synthesised.
    """
    round_ends = result.get("round_end_times_s") or []
    tokens_per_round = result.get("tokens_per_round") or []
    if not round_ends:
        return {
            "token_arrival_gaps_ms": [],
            "round_latencies_ms": [],
        }

    round_latencies_ms: List[float] = []
    prev = 0.0
    for t in round_ends:
        round_latencies_ms.append((t - prev) * 1000.0)
        prev = t

    arrival_times_s: List[float] = []
    for idx, t in enumerate(round_ends):
        n = tokens_per_round[idx] if idx < len(tokens_per_round) else 1
        arrival_times_s.extend([t] * n)

    gaps_ms: List[float] = []
    prev = 0.0
    for t in arrival_times_s:
        gaps_ms.append((t - prev) * 1000.0)
        prev = t

    return {
        "token_arrival_gaps_ms": gaps_ms,
        "round_latencies_ms": round_latencies_ms,
    }


def timed_generate(generate_fn: Callable, **kwargs) -> tuple[dict, float, dict]:
    """
    Run a generation function and measure wall-clock time.

    generate_fn must accept **kwargs (including `clock`) and return a dict with
    at least 'generated_ids'.

    Returns:
        (result_dict, wall_clock_s, latency_series)

    The device is synchronised before starting and after finishing so that
    wall_clock_s covers all queued work. Within the run, timestamps come from
    the generator itself at points where a .item() call has already forced a
    synchronisation, so instrumentation adds no extra barriers.
    """
    _synchronize()
    t0 = time.perf_counter()
    result = generate_fn(clock=time.perf_counter, **kwargs)
    _synchronize()
    t1 = time.perf_counter()
    wall_clock_s = t1 - t0
    return result, wall_clock_s, derive_latencies(result)


# ---------------------------------------------------------------------------
# Run record builder
# ---------------------------------------------------------------------------

def build_run_record(
    mode: str,
    target_model_name: str,
    draft_model_name: str,
    K: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    prompt_id: int,
    prompt_tokens: int,
    max_new_tokens: int,
    result: dict,
    wall_clock_s: float,
    latencies: dict,
    gpu_stats: Optional[dict] = None,
    arm: Optional[str] = None,
    arm_position: Optional[int] = None,
) -> dict:
    """Build the standard run record dict for appending to raw_measurements.jsonl."""
    generated_ids = result.get("generated_ids", [])
    n_generated = len(generated_ids)
    tokens_per_s = n_generated / wall_clock_s if wall_clock_s > 0 else 0.0

    accepted_per_round = result.get("accepted_per_round", [])
    tokens_per_round = result.get("tokens_per_round", [])
    # Acceptance length = tokens emitted per round. For the speculative path
    # this is accepted + 1 (the resampled or bonus token). Computed from the
    # recorded per-round emission counts rather than re-derived, so that EOS
    # truncation in the final round cannot inflate it.
    mean_acceptance_length = (
        float(np.mean(tokens_per_round)) if tokens_per_round else 0.0
    )
    mean_accepted_draft_tokens = (
        float(np.mean(accepted_per_round)) if accepted_per_round else 0.0
    )

    env = cuda_env_info()
    gpu = gpu_stats or {}

    record = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "arm": arm if arm is not None else mode,
        "arm_position": arm_position,
        "target_model": target_model_name,
        "draft_model": draft_model_name,
        "K": K,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "reproducible": result.get("reproducible", False),
        # Declared by the generator. Present only for paths that version
        # themselves (currently the EAGLE decoder); None elsewhere.
        "inference_version": result.get("inference_version"),
        "prompt_id": prompt_id,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": n_generated,
        "wall_clock_s": wall_clock_s,
        "tokens_per_s": tokens_per_s,
        "token_arrival_gaps_ms": latencies.get("token_arrival_gaps_ms", []),
        "round_latencies_ms": latencies.get("round_latencies_ms", []),
        "target_forward_calls": result.get("target_forward_calls", 0),
        "draft_forward_calls": result.get("draft_forward_calls", 0),
        "accepted_per_round": accepted_per_round,
        "tokens_per_round": tokens_per_round,
        "mean_acceptance_length": mean_acceptance_length,
        "mean_accepted_draft_tokens": mean_accepted_draft_tokens,
        "residual_fallback_count": result.get("residual_fallback_count", 0),
        "gpu_temp_c": gpu.get("gpu_temp_c"),
        "gpu_clock_mhz": gpu.get("gpu_clock_mhz"),
        "torch_version": env["torch_version"],
        "cuda_version": env["cuda_version"],
        "driver_version": env["driver_version"],
        "device": env["device"],
    }
    return record


# ---------------------------------------------------------------------------
# Prompt set
# ---------------------------------------------------------------------------

PROMPT_SET: List[str] = [
    # Prose prompts
    "Once upon a time in a small village nestled between two mountains,",
    "The history of the Roman Empire is one of the most fascinating subjects in",
    "Climate change is reshaping ecosystems around the world. Scientists have observed",
    "The philosophy of mind asks fundamental questions about consciousness and the",
    "In the year 2050, artificial intelligence had become so integrated into daily life that",
    "The Amazon rainforest is home to an estimated ten percent of all species on Earth.",
    "Renaissance art represents a turning point in Western cultural history, marked by",
    "The physics of black holes remains one of the most counterintuitive areas of",
    "Language acquisition in children follows predictable developmental milestones, beginning",
    "The economic theory of comparative advantage explains why international trade can benefit",
    "Deep in the ocean where sunlight cannot reach, ecosystems thrive around hydrothermal vents.",
    "The development of the internet transformed communication in ways that were",
    "Quantum mechanics describes the behavior of matter at the smallest scales, where",
    "The French Revolution of 1789 was driven by a complex interplay of financial crisis,",
    "Biodiversity loss accelerates as human activity encroaches on natural habitats.",
    # Code prompts
    "def binary_search(arr, target):\n    \"\"\"Search for target in sorted array.\"\"\"\n    ",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, data):\n        ",
    "import numpy as np\n\ndef matrix_multiply(A, B):\n    \"\"\"Multiply two matrices using numpy.\"\"\"\n    ",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    ",
    "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\n    def insert(self, value):\n        ",
    "def fibonacci(n, memo={}):\n    \"\"\"Return the nth Fibonacci number using memoization.\"\"\"\n    ",
    "import torch\nimport torch.nn as nn\n\nclass SimpleTransformer(nn.Module):\n    def __init__(self, vocab_size, d_model, nhead):\n        super().__init__()\n        ",
    "def parse_json_recursive(data, depth=0):\n    \"\"\"Recursively parse JSON and print structure.\"\"\"\n    ",
    "# Implement a thread-safe queue in Python\nimport threading\nfrom collections import deque\n\nclass ThreadSafeQueue:\n    def __init__(self):\n        ",
    "def merge_sorted_arrays(arr1, arr2):\n    \"\"\"Merge two sorted arrays into one sorted array.\"\"\"\n    ",
    # Mixed / instructional
    "Explain the difference between supervised and unsupervised machine learning:",
    "The key steps to implement gradient descent from scratch are:",
    "When designing a RESTful API, the most important principles to follow are:",
    "The main differences between TCP and UDP protocols are:",
    "To optimise a slow SQL query, the first steps to consider are:",
    "The SOLID principles in software engineering stand for:",
]

assert len(PROMPT_SET) >= 30, f"Prompt set has {len(PROMPT_SET)} entries; need >= 30."


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Interleaved multi-arm benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    """
    One measurement condition in an interleaved benchmark.

    name:              Unique identifier, used as the summary key.
    mode:              Coarse label recorded in each run record
                       ("baseline", "spec", "spec_eagle").
    fn:                Callable taking the generation kwargs (input_ids,
                       max_new_tokens, temperature, top_k, top_p, device, seed,
                       eos_token_id, clock) and returning a result dict.
    K:                 Speculation depth (0 for the baseline).
    draft_model_name:  Name of the drafter, recorded for provenance.
    is_baseline:       Exactly one arm must set this; speedups are computed
                       against it.
    """

    name: str
    mode: str
    fn: Callable
    K: int = 0
    draft_model_name: str = ""
    is_baseline: bool = False


def run_multiarm_benchmark(
    arms: List[Arm],
    tokenizer,
    target_model_name: str,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_new_tokens: int = 128,
    seed_base: int = 42,
    device: str = "cuda",
    n_warmup: int = WARMUP_RUNS,
    prompts: Optional[List[str]] = None,
    use_eos: bool = False,
    progress: bool = True,
) -> dict:
    """
    Run every arm on every prompt, interleaved, and summarise.

    For prompt i the arms are run in the order rotate(arms, i): arm j goes
    first on prompt j, so no arm systematically occupies the position that
    pays for a cold cache or benefits from a warm one. Every arm sees the same
    prompt and the same seed on that prompt, which is what makes the paired
    bootstrap in bootstrap_speedup_ci valid.

    Warmup: results from the first `n_warmup` prompts are discarded, but the
    generations still run so that CUDA kernels, autotuning and clocks settle.

    `use_eos` is off by default: stopping early on EOS makes generation lengths
    differ between arms and prompts, which adds variance to a throughput
    comparison for no benefit. The correctness tests do pass an EOS id.

    Returns a dict of {arm_name: summary}, each summary containing the
    speedup against the baseline arm with a 95% bootstrap CI.
    """
    prompts = prompts if prompts is not None else PROMPT_SET
    baseline_arms = [a for a in arms if a.is_baseline]
    if len(baseline_arms) != 1:
        raise ValueError(f"Exactly one arm must be the baseline; got {len(baseline_arms)}.")
    baseline_arm = baseline_arms[0]

    eos_id = tokenizer.eos_token_id if use_eos else None
    records: Dict[str, List[dict]] = {a.name: [] for a in arms}

    for prompt_idx, prompt_text in enumerate(prompts):
        seed = seed_base + prompt_idx
        is_warmup = prompt_idx < n_warmup

        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        prompt_tokens = int(input_ids.shape[1])

        # Rotate the arm order so position within the prompt is not confounded
        # with arm identity.
        rotation = prompt_idx % len(arms)
        ordered = arms[rotation:] + arms[:rotation]

        for position, arm in enumerate(ordered):
            gpu_stats = get_gpu_stats()
            result, wall_clock_s, latencies = timed_generate(
                arm.fn,
                input_ids=input_ids.clone(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
                seed=seed,
                eos_token_id=eos_id,
            )
            if is_warmup:
                continue
            rec = build_run_record(
                mode=arm.mode,
                target_model_name=target_model_name,
                draft_model_name=arm.draft_model_name,
                K=arm.K,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                prompt_id=prompt_idx,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                result=result,
                wall_clock_s=wall_clock_s,
                latencies=latencies,
                gpu_stats=gpu_stats,
                arm=arm.name,
                arm_position=position,
            )
            records[arm.name].append(rec)
            _append_record(rec)

        if progress and not is_warmup:
            done = len(records[baseline_arm.name])
            print(
                f"    prompt {prompt_idx + 1}/{len(prompts)} "
                f"({done} scored) baseline "
                f"{records[baseline_arm.name][-1]['tokens_per_s']:.1f} tok/s",
                flush=True,
            )

    baseline_records = records[baseline_arm.name]
    if not baseline_records:
        return {"error": "No records collected (were all prompts warmup?)."}

    baseline_tps = np.array([r["tokens_per_s"] for r in baseline_records])
    summaries: Dict[str, dict] = {}

    for arm in arms:
        arm_records = records[arm.name]
        if not arm_records:
            continue
        arm_tps = np.array([r["tokens_per_s"] for r in arm_records])
        rng = np.random.default_rng(seed_base)
        speedup, ci_low, ci_high = bootstrap_speedup_ci(baseline_tps, arm_tps, rng=rng)

        acc_lengths = np.array([r["mean_acceptance_length"] for r in arm_records])
        if acc_lengths.size and acc_lengths.any():
            alpha_mean, alpha_lo, alpha_hi = bootstrap_mean_ci(acc_lengths, rng=rng)
        else:
            alpha_mean = alpha_lo = alpha_hi = 0.0

        gaps = np.concatenate(
            [np.asarray(r["token_arrival_gaps_ms"]) for r in arm_records]
        ) if any(r["token_arrival_gaps_ms"] for r in arm_records) else np.array([])
        rounds = np.concatenate(
            [np.asarray(r["round_latencies_ms"]) for r in arm_records]
        ) if any(r["round_latencies_ms"] for r in arm_records) else np.array([])

        summaries[arm.name] = {
            "arm": arm.name,
            "mode": arm.mode,
            "K": arm.K,
            "target_model": target_model_name,
            "draft_model": arm.draft_model_name,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "n_prompts": len(arm_records),
            "baseline_mean_tps": float(baseline_tps.mean()),
            "mean_tps": float(arm_tps.mean()),
            "speedup": speedup,
            "speedup_ci_95_low": ci_low,
            "speedup_ci_95_high": ci_high,
            "speedup_str": f"{speedup:.2f} (95% CI {ci_low:.2f}-{ci_high:.2f})",
            "mean_acceptance_length": alpha_mean,
            "acceptance_length_ci_95_low": alpha_lo,
            "acceptance_length_ci_95_high": alpha_hi,
            "mean_accepted_draft_tokens": float(
                np.mean([r["mean_accepted_draft_tokens"] for r in arm_records])
            ),
            "mean_target_forward_calls": float(
                np.mean([r["target_forward_calls"] for r in arm_records])
            ),
            "mean_draft_forward_calls": float(
                np.mean([r["draft_forward_calls"] for r in arm_records])
            ),
            "residual_fallback_total": int(
                sum(r["residual_fallback_count"] for r in arm_records)
            ),
            "token_arrival_gap_percentiles": (
                latency_percentiles(gaps) if gaps.size else {}
            ),
            "round_latency_percentiles": (
                latency_percentiles(rounds) if rounds.size else {}
            ),
            "gpu_temp_c_first": arm_records[0].get("gpu_temp_c"),
            "gpu_temp_c_last": arm_records[-1].get("gpu_temp_c"),
        }

    return summaries


def run_interleaved_benchmark(
    target_model: torch.nn.Module,
    draft_model: Optional[torch.nn.Module],
    tokenizer,
    target_model_name: str,
    draft_model_name: str,
    K: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_new_tokens: int = 128,
    seed_base: int = 42,
    device: str = "cuda",
    n_warmup: int = WARMUP_RUNS,
    mode: str = "spec",
    generate_baseline_fn: Optional[Callable] = None,
    generate_spec_fn: Optional[Callable] = None,
    tag: Optional[str] = None,
) -> dict:
    """
    Two-arm convenience wrapper over run_multiarm_benchmark.

    Kept for callers that only want baseline vs. one speculative configuration.
    The result is the speculative arm's summary, with the baseline summary
    nested under "baseline".
    """
    from specbench.baseline import generate_baseline
    from specbench.spec_decode import generate_speculative

    if generate_baseline_fn is None:
        def generate_baseline_fn(**kw):
            return generate_baseline(target_model, **kw)

    if generate_spec_fn is None:
        def generate_spec_fn(**kw):
            return generate_speculative(draft_model, target_model, K=K, **kw)

    arms = [
        Arm(name="baseline", mode="baseline", fn=generate_baseline_fn,
            K=0, draft_model_name="", is_baseline=True),
        Arm(name=f"{mode}_K{K}", mode=mode, fn=generate_spec_fn,
            K=K, draft_model_name=draft_model_name),
    ]
    summaries = run_multiarm_benchmark(
        arms=arms,
        tokenizer=tokenizer,
        target_model_name=target_model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed_base=seed_base,
        device=device,
        n_warmup=n_warmup,
    )
    if "error" in summaries:
        return summaries
    result = summaries[f"{mode}_K{K}"]
    result["baseline"] = summaries["baseline"]
    merge_summary(tag or f"{mode}_K{K}", result)
    return result
