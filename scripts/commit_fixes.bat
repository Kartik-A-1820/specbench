@echo off
REM ====================================================================
REM  scripts\commit_fixes.bat - commit the post-initial-history work.
REM
REM  scripts\setup_repo.bat lays down the initial staged commits. This
REM  script commits everything done since, in meaningful groups rather
REM  than one lump. Safe to re-run: a group with nothing staged is
REM  skipped rather than erroring.
REM ====================================================================
setlocal
cd /d "%~dp0.."

where git >nul 2>&1
if errorlevel 1 (
    echo [commit_fixes] git is not on PATH.
    exit /b 1
)
if exist ".git\index.lock" del /f /q ".git\index.lock"

echo [commit_fixes] Committing environment and runner work...
git add run_all.bat rerun_vram_probe.bat scripts\setup_env.bat scripts\check_env.py ^
        scripts\report_status.py scripts\setup_repo.bat scripts\commit_fixes.bat requirements.txt .gitignore
git diff --cached --quiet || git commit -q -m "Self-provisioning batch runner with resumable stages

Builds a project-local .venv with a CUDA build of torch rather than
touching the system interpreter, which carries a CPU-only wheel.
check_env.py refuses to proceed on a CPU-only torch, which otherwise runs
every stage to completion and produces meaningless numbers. Console
forced to UTF-8: the default cp1252 code page turns any non-ASCII
character in a print statement into a stage-killing UnicodeEncodeError."

echo [commit_fixes] Committing the EAGLE causal-mask fix...
git add specbench\eagle\head.py tests\test_eagle_head.py scripts\run_phase2_train.py scripts\run_phase2_sweep.py
git diff --cached --quiet || git commit -q -m "Fix bidirectional attention in the EAGLE draft head

A bare LlamaDecoderLayer built from a plain LlamaConfig uses the eager
attention path, which applies no mask at all when attention_mask is None.
Attention was therefore bidirectional during training, and since the label
for position t is the input token at position t+1, the head read its
answer off the future. Final cross-entropy came out at 0.053 against a
floor of 2.98 - the target model's own cross-entropy on the same cached
features, which a distilled head cannot beat.

Nothing failed visibly: training looked excellent and the only symptom
would have been an EAGLE benchmark showing no speedup, which reads as a
legitimate negative result.

- forward() now builds an explicit causal mask, independent of which
  attention implementation the config selects.
- tests/test_eagle_head.py asserts outputs at t are unaffected by inputs
  at positions > t. Verified to fail without the fix.
- Training warns below a plausibility floor on the loss; the sweep refuses
  to benchmark a flagged head, or one whose ARCH_VERSION does not match."

echo [commit_fixes] Committing corpus and data-integrity work...
git add specbench\eagle\collect.py scripts\run_phase2_collect.py specbench\eagle\train.py
git diff --cached --quiet || git commit -q -m "Train the EAGLE head on a real corpus

The previous corpus was 13 hand-written sentences repeated 40 times,
labelled as ~500K tokens. A head fitted to that memorises 13 sentences,
so any acceptance rate measured against it says nothing about the method.
Now wikitext-2 plus mbpp, with per-source provenance recorded. If no
source loads, the run is marked degenerate and Phase 2 is not reported."

echo [commit_fixes] Committing measurement and reporting work...
git add specbench\models.py scripts\run_vram_probe.py scripts\make_pareto.py ^
        scripts\make_results_doc.py scripts\download_models.py
git diff --cached --quiet || git commit -q -m "Measurement corrections and generated results

- VRAM probe no longer attributes the retained prefill logits tensor to
  the KV cache. At 2048 context that tensor is ~131 MB, roughly 3x the KV
  cache itself, and counting it inflated the KV line by the same factor.
- Draft tokenizer is not loaded: the two models share the target's
  vocabulary by necessity, and llama-68m ships a SentencePiece-only
  tokenizer whose conversion needs protobuf.
- Pareto aggregation keys on temperature, so greedy and sampling
  baselines are no longer pooled into one comparison.
- docs/results.md is generated from the raw measurements; unmeasured
  configurations print NOT MEASURED instead of a plausible number."

echo [commit_fixes] Committing documentation...
git add docs README.md
git diff --cached --quiet || git commit -q -m "Documentation: measured results, honest status, KT source

Removes claims the project had not earned - a README describing itself as
benchmarked before any benchmark had run, limitations written in the past
tense about measurements not yet taken, and a reference to a loss-curve
figure that did not exist."

echo [commit_fixes] Committing remaining files...
git add -A
git diff --cached --quiet || git commit -q -m "Results, figures and run artifacts"

echo.
echo [commit_fixes] Log:
git --no-pager log --oneline
echo.
echo Press any key to close.
pause >nul
exit /b 0
