# specbench

**A reproducible study of speculative decoding under a small GPU memory budget.**

Speculative decoding uses a cheaper draft model to propose tokens, then lets the target model verify them. This repository implements the decoding loop, KV-cache rollback, correctness checks, paired benchmarks, and a trained EAGLE-style draft head. It is intended for readers who want to inspect the tradeoffs and reproduce the measurements.

## Published result

| Experiment | Measured speedup | 95% bootstrap interval |
| --- | --- | --- |
| Separate draft model, sampling, K=3 | **1.321×** | 1.235–1.406 |
| Separate draft model, greedy, K=2 | **1.442×** | 1.373–1.510 |
| Best reported EAGLE-style head configuration | **0.938×** | 0.921–0.956 |

These are repository measurements on one **GTX 1650 Ti 4 GB** laptop, using TinyLlama-1.1B as the target and llama-68m as the separate draft. The paired benchmark uses 29 scored prompts and 128 generated tokens per prompt. The trained-head result is a slowdown, retained with its diagnosis.

[Full results](docs/results.md) · [Methodology](docs/methodology.md) · [Limitations](docs/limitations.md) · [Raw records](results/raw_measurements.jsonl)

Hardware, model pair, prompt distribution, and decoding settings affect acceptance and throughput. These figures are not universal speedups.

## What the implementation demonstrates

- Draft/verify decoding with residual sampling and shared sampling policy.
- KV-cache handling and rollback after rejected proposals.
- Interleaved baseline/speculative measurements to reduce ordering and thermal bias.
- Bootstrap uncertainty, latency breakdowns, and acceptance/cost analysis.
- A trained draft-head experiment whose negative result exposes the quality/cost tradeoff.

Under the algorithm's assumptions, speculative sampling preserves the target distribution. The synthetic tests provide empirical checks of this implementation; a Monte Carlo test is not a mathematical proof of the entire codebase.

## Start with the right workload

| Goal | Requirements | Starting point |
| --- | --- | --- |
| Read the study | Browser only | Results and methodology above |
| Run synthetic correctness checks | Python, CPU PyTorch, project dependencies | CPU path below; no model weights |
| Reproduce model speedups | Compatible NVIDIA GPU/CUDA, downloaded models | GPU path below |
| Train/sweep draft heads | GPU environment, training data, disk for hidden-state cache | Phase 2 scripts and methodology |

## Environment setup

Use a separate environment. The published environment used **PyTorch 2.5.1+cu121**; other versions/hardware are new environments to validate, not equivalent benchmark conditions.

```bash
git clone https://github.com/Kartik-A-1820/specbench.git
cd specbench
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Run all commands from the repository root.

**CPU correctness work:**

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest tests/test_sampler_synthetic.py -v
```

The synthetic suite uses repeated Monte Carlo trials and can take time. It does not download models or measure GPU speed.

**GPU benchmark work:** in a separate environment, install the published CUDA build instead of the CPU build:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/check_env.py
python scripts/download_models.py
python scripts/run_vram_probe.py
python -m pytest tests/ -v -m "not slow"
python scripts/run_phase1_bench.py --k-values 1 2 3 4 6 8 --temperature 1.0
```

The environment/probe checks enforce the project's GPU assumptions; a different GPU may need deliberate changes before it can run the same pipeline. First downloads require network access. Subsequent benchmark execution uses local model artifacts.

`run_all.bat` is a Windows convenience workflow that provisions an environment, downloads models, runs checks, and executes benchmark stages. It uses stage markers to skip completed work. Inspect those markers and result files before interpreting a rerun as a fresh experiment; keep prior results when comparing configurations.

## Results and reproduction

`scripts/make_results_doc.py` regenerates `docs/results.md` from stored measurements. Edit experiment configuration and rerun measurements rather than typing numbers into generated tables. Unmeasured configurations should remain labeled as such.

The slower end-to-end distribution test is marked `slow` and requires real models. Passing synthetic checks alone does not establish every generation path's correctness. Keep model revisions, settings, environment details, raw records, and run provenance with any new result.

## Contribute

[CONTRIBUTING.md](CONTRIBUTING.md) describes the review and reproduction workflow. Useful contributions include:

- Reproductions on different hardware with complete environment records.
- Sampler/cache edge-case tests and small correctness fixes.
- Broader held-out prompt coverage and transparent draft-head training experiments.
- Documentation of setup failures, version constraints, and profiler findings.

There is currently no standalone LICENSE file. Clarify intended reuse terms with the maintainer; external model and dataset licenses are separate.

## Repository map

| Path | Role |
| --- | --- |
| `specbench/` | Baseline/speculative decoding, sampling, KV logic, statistics |
| `specbench/eagle/` | Draft-head collection, training, decoding |
| `scripts/` | Setup, environment checks, benchmark/report entrypoints |
| `tests/` | Synthetic and real-model correctness checks |
| `results/` | Measured summaries, raw records, figures |
| `docs/` | Architecture, methodology, results, limitations |
