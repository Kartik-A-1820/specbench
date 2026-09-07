@echo off
REM ===================================================================
REM  scripts\setup_repo.bat - one-time repository hygiene.
REM
REM  Removes the build-time scratch documents, initialises git if the
REM  project is not yet a repository, and makes a set of staged commits.
REM  Safe to re-run: it does nothing if commits already exist.
REM ===================================================================
setlocal
cd /d "%~dp0.."

echo [setup_repo] Working directory: %CD%

REM --- Remove build scratch files that should not ship ---------------
if exist "BUILD_BRIEF.md" (
    del /f /q "BUILD_BRIEF.md"
    echo [setup_repo] Deleted BUILD_BRIEF.md
)
if exist "create.md" (
    del /f /q "create.md"
    echo [setup_repo] Deleted create.md
)

REM --- Clear a stale index lock left by a crashed git process ---------
if exist ".git\index.lock" (
    del /f /q ".git\index.lock"
    echo [setup_repo] Removed stale .git\index.lock
)

where git >nul 2>&1
if errorlevel 1 (
    echo [setup_repo] git is not on PATH. Skipping repository setup.
    exit /b 0
)

if not exist ".git" (
    git init -q
    echo [setup_repo] Initialised a new git repository.
)

git rev-parse --verify HEAD >nul 2>&1
if not errorlevel 1 (
    echo [setup_repo] Repository already has commits. Skipping initial commits.
    exit /b 0
)

git config user.name  >nul 2>&1 || git config user.name  "specbench"
git config user.email >nul 2>&1 || git config user.email "specbench@localhost"

echo [setup_repo] Creating staged commits...

git add .gitignore requirements.txt pytest.ini
git commit -q -m "Project scaffolding: dependencies, pytest config, gitignore"

git add specbench/__init__.py specbench/models.py specbench/kv.py specbench/sampling.py
git commit -q -m "Model loading, KV cache handling with rollback, and logits processors"

git add specbench/spec_decode.py specbench/baseline.py
git commit -q -m "Speculative decoding core and target-only baseline

Single target forward pass per round: the pending token from the previous
round heads the verification block, so one pass over [pending, x_1..x_K]
yields p_1..p_{K+1}."

git add specbench/stats.py specbench/harness.py
git commit -q -m "Benchmark harness: interleaved multi-arm design, bootstrap CIs, real per-round latency instrumentation"

git add specbench/eagle
git commit -q -m "EAGLE draft head: module, offline hidden-state collection, training loop, decoding path"

git add tests
git commit -q -m "Correctness tests: synthetic sampler, greedy equivalence, KV rollback, VRAM budget"

git add scripts
git commit -q -m "Entry points: model download, VRAM probe, Phase 1 benchmark, Phase 2 collect/train/sweep, figures"

git add docs README.md
git commit -q -m "Documentation: architecture, methodology, VRAM budget, limitations, results"

git add -A
git commit -q -m "Batch runner and remaining project files" 2>nul

echo [setup_repo] Done. Commit log:
git --no-pager log --oneline
exit /b 0
