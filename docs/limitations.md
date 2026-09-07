# Limitations

This document records the known limitations of the specbench implementation
and measurements. It is intentionally specific and honest.

---

## Hardware and Environment

**Single-device results.** Measurements are taken on one GTX 1650 Ti Mobile
(SM 7.5, 3.4 GB usable VRAM); the device name, capability, driver and torch
build are recorded in every row of `results/raw_measurements.jsonl`. Results
may not transfer to other hardware. The same K-value that maximises speedup on this GPU may not be
optimal on Ampere or Ada hardware with larger VRAM and different bandwidth.

**Thermal variance.** The GTX 1650 Ti in a laptop chassis throttles
aggressively under sustained load. GPU clock during benchmarks varies
depending on ambient temperature and how recently the GPU was idle. The
interleaved A/B design mitigates per-prompt thermal bias (see
`docs/methodology.md`), but run-to-run variance from session-to-session
thermal state is not controlled. Reported confidence intervals reflect
within-session variance only.

**No cross-hardware validation.** The VRAM arithmetic and model selection
were validated on this specific machine. A different GPU with different WDDM
overhead, different driver behaviour, or a different Windows version may have
a different usable headroom.

---

## Benchmark Scope

**Batch size 1 only.** The harness generates one sequence at a time.
Speculative decoding in batch-size-B settings involves different dynamics
(the draft must propose tokens that all B sequences accept; acceptance rate
drops with batch size). Results do not apply to batched serving.

**No serving stack integration.** The benchmark runs inference in a standalone
Python process, not inside a serving framework (vLLM, TGI, etc.). Real
production speedups depend on the serving implementation.

**Small prompt set.** The 30-prompt benchmark set is large enough for
statistically meaningful comparisons but too small to cover all natural language
domains, languages, or task types. Acceptance rate depends heavily on topic and
register.

**Max new tokens = 128.** Short generation windows may not fully represent
the expected distribution of output lengths in real usage.

---

## Correctness Tests

**Status as of implementation:** The synthetic sampler tests
(`test_sampler_synthetic.py`) run on CPU without model loading and do not depend
on the hardware. The end-to-end tests (`test_greedy_equivalence.py`,
`test_distribution_match.py`, `test_kv_rollback.py`, `test_vram_budget.py`)
require CUDA and the downloaded models.

> If any test is not currently green on this machine, the specific failure will
> be noted here. Until the models are downloaded and the environment is set up,
> the GPU-dependent tests cannot be run.

**`test_distribution_match.py` is marked `@pytest.mark.slow`** and requires
N=20,000 repeated generation passes with real models. It is not run in quick
`pytest` invocations (`pytest -m "not slow"`).

---

## Phase 2 (EAGLE)

**Training data.** The built-in training corpus in `specbench/eagle/collect.py`
is a small repeated set of ~40 documents. This is sufficient to validate the
architecture but likely too small and homogeneous to achieve maximum acceptance
rates. A larger and more diverse corpus (e.g., a slice of C4 or the Pile) would
be expected to improve acceptance rates.

**Offline collection latency.** Hidden states are cached at 4 KB per token
(2048 dims, FP16), so the corpus size is bounded by collection time and disk
rather than by what would train a good head. The default cap is 250 K token
positions, about 1 GB. That is small for this purpose: EAGLE heads in the
literature are trained on orders of magnitude more.

**Training corpus.** The head is trained on wikitext-2 plus mbpp — general
prose and short Python. The benchmark prompt set contains code idioms and
instructional phrasing that are underrepresented there, so measured acceptance
is a floor, not a ceiling, for what this head architecture can reach. If
neither dataset can be downloaded, `collect_hidden_states` flags the run as
degenerate and Phase 2 is not reported at all.

**Head training convergence.** Three epochs on a 250 K-token cache may not
reach convergence. Per-head loss curves are written to
`results/figures/<head_name>_loss_curves.png` when training runs; if they have
not plateaued, more epochs are warranted. Those figures do not exist until
training has actually run.

---

## Numbers Not Yet Available

The following measurements are pending and will be filled in after running the
benchmark scripts:

- Phase 0: measured VRAM allocation vs. prediction (see `results/vram_probe.json`).
- Phase 1: speedup by K, latency percentiles, acceptance rate by domain
  (see `docs/results.md`).
- Phase 2: EAGLE sweep results and Pareto curve (see `results/pareto.csv`).

No numbers have been fabricated. All reported values will trace directly to
rows in `results/raw_measurements.jsonl`.
