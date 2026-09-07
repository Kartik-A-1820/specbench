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

REM --- Timestamped log file ------------------------------------------
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set LDT=%%I
if "%LDT%"=="" (
    set STAMP=manual
) else (
    set STAMP=!LDT:~0,8!_!LDT:~8,4!
)
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

if exist "%STAGES%\phase2_train.done" (
    echo [skip] head training already done
) else (
    %PY% scripts\run_phase2_train.py --epochs 3 --done-marker "%STAGES%\phase2_train.done"
    if errorlevel 1 (
        echo [WARN] EAGLE head training failed. Skipping the Phase 2 sweep.
        goto :STAGE8_SKIP
    )
)

if exist "%STAGES%\phase2_sweep.done" (
    echo [skip] sweep already done
) else (
    %PY% scripts\run_phase2_sweep.py --k-values 1 2 3 4 6 8 ^
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
echo === STAGE 9/9: figures ===
REM ====================================================================
%PY% scripts\make_pareto.py
if errorlevel 1 (
    echo [WARN] Figure generation failed. Numeric results are unaffected.
)
echo [STAGE 9 OK]
echo.

echo ############################################################
echo # ALL STAGES ATTEMPTED
echo # finished: %DATE% %TIME%
echo ############################################################
%PY% scripts\report_status.py
exit /b 0
