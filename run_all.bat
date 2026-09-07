@echo off
REM ====================================================================
REM  run_all.bat - end-to-end specbench run.
REM
REM  Double-click this file, or run it from a terminal. It is entirely
REM  non-interactive: no prompts, no keypresses needed until the very end.
REM
REM  Everything is written to logs\run_<timestamp>.log. Each stage prints
REM  a marker line, so a partial log tells you exactly where it stopped.
REM
REM  RESUMABLE. Every stage checks for its own output artifact and skips
REM  if it is already there. If the machine sleeps, thermally throttles
REM  into oblivion, or you close the window, just run it again.
REM
REM  To force a stage to re-run, delete its marker in logs\stages\ (and,
REM  for benchmarks, the corresponding results\*_summary.json).
REM ====================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM --- Console and Python I/O encoding --------------------------------
REM  The Windows console defaults to cp1252 here, and Python inherits it for
REM  stdout. Any non-ASCII character in a print statement then raises
REM  UnicodeEncodeError and kills the stage. Force UTF-8, and fall back to
REM  backslash escapes rather than an exception if anything still cannot be
REM  encoded.
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8:backslashreplace
set PYTHONUTF8=1

REM  Unbuffered stdout. Python block-buffers stdout when it is redirected to a
REM  file, so a long stage can appear frozen in the log for minutes while it is
REM  in fact working. That makes a partial log much harder to diagnose, which is
REM  the whole point of writing one.
set PYTHONUNBUFFERED=1

REM --- Timestamped log file ------------------------------------------
REM  wmic is absent on recent Windows 11 builds, which silently collapsed every
REM  run onto one log file called run_manual.log. Two concurrent runs then
REM  truncate each other's log, and a re-run destroys the record of the run
REM  being diagnosed. PowerShell is already a dependency further down, so use
REM  it; wmic stays as a fallback for older machines.
set STAMP=
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" 2^>nul') do set STAMP=%%I
if "%STAMP%"=="" (
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set LDT=%%I
    if not "!LDT!"=="" set STAMP=!LDT:~0,8!_!LDT:~8,4!
)
if "%STAMP%"=="" set STAMP=manual
if not exist "logs" mkdir "logs"
if not exist "logs\stages" mkdir "logs\stages"
set LOG=logs\run_%STAMP%.log
set STAGES=logs\stages

echo Writing to %LOG%
echo Close this window at any time; re-run run_all.bat to resume.

call :RUN_ALL > "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   RUN COMPLETE. Full log: %LOG%
) else (
    echo   RUN STOPPED with code %RC%. Full log: %LOG%
    echo   Re-run run_all.bat to resume from the last completed stage.
)
echo ============================================================
echo.
echo Last 40 lines of the log:
echo ------------------------------------------------------------
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 40" 2>nul
echo ------------------------------------------------------------
echo.
echo Press any key to close.
pause >nul
exit /b %RC%


REM ====================================================================
:RUN_ALL
REM ====================================================================
echo ############################################################
echo # specbench full run
echo # started: %DATE% %TIME%
echo # directory: %CD%
echo ############################################################
echo.

REM ====================================================================
echo === STAGE 0/9: environment provisioning ===
REM
REM  The system interpreter on this machine carries a CPU-only torch. Rather
REM  than replace it globally and risk breaking other projects, specbench
REM  builds its own .venv with a CUDA build of torch. Created once, then
REM  reused on every later run.
REM ====================================================================
if exist ".venv\Scripts\python.exe" (
    echo [skip] .venv already exists
) else (
    call scripts\setup_env.bat
    if errorlevel 1 (
        echo [FATAL] Could not provision the environment. See above.
        exit /b 91
    )
)

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [FATAL] .venv\Scripts\python.exe is missing after provisioning.
    exit /b 92
)
echo [env] Using interpreter: %PY%
%PY% -c "import sys; print('[env] python', sys.version)"
echo [STAGE 0 OK]
echo.

REM ====================================================================
echo === STAGE 1/9: environment and CUDA assertion ===
REM ====================================================================
%PY% scripts\check_env.py
if errorlevel 1 (
    echo [FATAL] Environment check failed. See the message above.
    echo         If torch is CPU-only, delete the .venv folder and re-run
    echo         run_all.bat to rebuild it from the CUDA wheel index.
    exit /b 1
)
echo [STAGE 1 OK]
echo.

REM ====================================================================
echo === STAGE 2/9: repository hygiene and git ===
REM ====================================================================
if exist "%STAGES%\repo.done" (
    echo [skip] already done
) else (
    call scripts\setup_repo.bat
    echo ok> "%STAGES%\repo.done"
)
echo [STAGE 2 OK]
echo.

REM ====================================================================
echo === STAGE 3/9: download models ===
REM ====================================================================
if exist "models\TinyLlama-1.1B-Chat-v1.0\config.json" if exist "models\llama-68m\config.json" (
    echo [skip] models already present
    goto :STAGE3_DONE
)
%PY% scripts\download_models.py
if errorlevel 1 (
    echo [FATAL] Model download failed. Check network access to huggingface.co.
    exit /b 3
)
:STAGE3_DONE
echo [STAGE 3 OK]
echo.

REM ====================================================================
echo === STAGE 4/9: CPU-only correctness tests (synthetic sampler, stats) ===
REM ====================================================================
if exist "%STAGES%\tests_cpu.done" (
    echo [skip] already done
) else (
    REM Head structure tests run first: they take seconds, need no GPU, and
    REM catch the causal-mask leak that makes EAGLE training silently useless.
    %PY% -m pytest tests\test_eagle_head.py -q
    if errorlevel 1 (
        echo [FATAL] EAGLE head structural tests failed. Attention inside the head
        echo         is probably not causal, which makes any trained head worthless.
        exit /b 4
    )
    %PY% -m pytest tests\test_sampler_synthetic.py -q
    if errorlevel 1 (
        echo [FATAL] The synthetic sampler tests failed. The accept/reject rule is
        echo         wrong or the environment is broken. Do not benchmark until
        echo         this passes.
        exit /b 4
    )
    echo ok> "%STAGES%\tests_cpu.done"
)
echo [STAGE 4 OK]
echo.

REM ====================================================================
echo === STAGE 5/9: VRAM probe ===
REM ====================================================================
if exist "%STAGES%\vram.done" (
    echo [skip] already done
) else (
    %PY% scripts\run_vram_probe.py
    if errorlevel 1 (
        echo [FATAL] VRAM probe failed or exceeded the budget. Retry with a
        echo         shorter context: %PY% scripts\run_vram_probe.py --context-length 1024
        exit /b 5
    )
    echo ok> "%STAGES%\vram.done"
)
echo [STAGE 5 OK]
echo.

REM ====================================================================
echo === STAGE 6/9: GPU correctness tests (greedy equivalence, KV rollback) ===
REM ====================================================================
if exist "%STAGES%\tests_gpu.done" (
    echo [skip] already done
) else (
    %PY% -m pytest tests\test_kv_rollback.py tests\test_greedy_equivalence.py tests\test_vram_budget.py -q
    if errorlevel 1 (
        echo [FATAL] GPU correctness tests failed. Benchmarking a wrong
        echo         implementation produces numbers that mean nothing. Stopping.
        exit /b 6
    )
    echo ok> "%STAGES%\tests_gpu.done"
)
echo [STAGE 6 OK]
echo.

REM ====================================================================
echo === STAGE 7/9: Phase 1 benchmark (draft model, K sweep) ===
REM ====================================================================
if exist "%STAGES%\phase1.done" (
    echo [skip] already done
) else (
    echo [phase1] sampling run, temperature 1.0
    %PY% scripts\run_phase1_bench.py --temperature 1.0 --k-values 1 2 3 4 6 8 ^
        --max-new-tokens 128 --tag phase1_T1.0 --done-marker "%STAGES%\phase1_t1.done"
    if errorlevel 1 (
        echo [FATAL] Phase 1 sampling benchmark failed.
        exit /b 7
    )
    echo [phase1] greedy run, temperature 0.0
    %PY% scripts\run_phase1_bench.py --temperature 0.0 --k-values 1 2 4 8 ^
        --max-new-tokens 128 --tag phase1_T0.0 --done-marker "%STAGES%\phase1_t0.done"
    if errorlevel 1 (
        echo [FATAL] Phase 1 greedy benchmark failed.
        exit /b 7
    )
    echo ok> "%STAGES%\phase1.done"
)
echo [STAGE 7 OK]
echo.

REM ====================================================================
echo === STAGE 8/9: Phase 2 EAGLE (collect, train, sweep) ===
REM ====================================================================
if exist "%STAGES%\phase2_collect.done" (
    echo [skip] hidden-state collection already done
) else (
    %PY% scripts\run_phase2_collect.py --done-marker "%STAGES%\phase2_collect.done"
    if errorlevel 1 (
        echo [WARN] Hidden-state collection failed. Phase 2 will be skipped and
        echo        Phase 1 results stand on their own.
        goto :STAGE8_SKIP
    )
)

REM  No marker gate on training. The marker records "the training stage ran",
REM  which is coarser than "every head is current" — so after a fix to the head
REM  architecture the marker would skip training while the checkpoints on disk
REM  are still stale, and the sweep would then refuse and the run would
REM  dead-end. run_phase2_train.py does the right check itself, comparing each
REM  checkpoint against ARCH_VERSION, and is a fast no-op when nothing needs
REM  retraining. Let the script decide.
%PY% scripts\run_phase2_train.py --epochs 3 --done-marker "%STAGES%\phase2_train.done"
if errorlevel 1 (
    echo [WARN] EAGLE head training failed. Skipping the Phase 2 sweep.
    goto :STAGE8_SKIP
)

if exist "%STAGES%\phase2_sweep.done" (
    echo [skip] sweep already done
) else (
    REM K is swept over {1,2,3,4,6} rather than out to 8. Phase 1 measured the
    REM optimum at K=3 with K=8 already down at 1.10x and its CI touching 1.0,
    REM so K=8 costs four more head-configurations' worth of runtime to confirm
    REM a decline that is already established. Five arms x four heads is the
    REM most informative sweep that fits in the time budget.
    %PY% scripts\run_phase2_sweep.py --k-values 1 2 3 4 6 ^
        --max-new-tokens 128 --done-marker "%STAGES%\phase2_sweep.done"
    if errorlevel 1 (
        echo [WARN] Phase 2 sweep failed or was interrupted. Re-running run_all.bat
        echo        resumes at the next untested head.
        goto :STAGE8_SKIP
    )
)
echo [STAGE 8 OK]
goto :STAGE8_END
:STAGE8_SKIP
echo [STAGE 8 INCOMPLETE - Phase 2 results are not available]
:STAGE8_END
echo.

REM ====================================================================
echo === STAGE 9/9: figures and generated results document ===
REM ====================================================================
%PY% scripts\make_pareto.py
if errorlevel 1 (
    echo [WARN] Figure generation failed. Numeric results are unaffected.
)
%PY% scripts\make_results_doc.py
if errorlevel 1 (
    echo [WARN] Could not regenerate docs\results.md.
)
call scripts\commit_fixes.bat < nul
echo [STAGE 9 OK]
echo.

echo ############################################################
echo # ALL STAGES ATTEMPTED
echo # finished: %DATE% %TIME%
echo ############################################################
%PY% scripts\report_status.py
exit /b 0
