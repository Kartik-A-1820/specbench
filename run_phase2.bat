@echo off
REM ====================================================================
REM  run_phase2.bat - finish Phase 2 and the reporting.
REM
REM  Self-contained and independent of run_all.bat. Phase 0/1 are already
REM  complete and recorded in results\; this script does only what is
REM  left:
REM
REM    1. EAGLE head training (retrains any head whose ARCH_VERSION is
REM       stale - all four are, after the causal-mask fix)
REM    2. EAGLE benchmark sweep
REM    3. Figures and the generated results document
REM    4. Git commits
REM
REM  Writes to its own timestamped log so it cannot collide with a log
REM  file another console still holds open.
REM
REM  Resumable: training skips heads already at the current architecture,
REM  and the sweep skips heads whose summary already exists. If this is
REM  interrupted, just run it again.
REM ====================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8:backslashreplace
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

set STAMP=
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" 2^>nul') do set STAMP=%%I
if "%STAMP%"=="" set STAMP=p2

if not exist "logs" mkdir "logs"
set LOG=logs\phase2_%STAMP%.log

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [FATAL] %PY% not found. Run run_all.bat first to provision the environment.
    echo Press any key to close.
    pause >nul
    exit /b 90
)

echo Writing to %LOG%
echo This runs for roughly 3 hours. Close the window at any time and
echo re-run this file to resume; nothing is lost.

call :PHASE2 > "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo.
echo ============================================================
if "%RC%"=="0" (
    echo   PHASE 2 COMPLETE. Log: %LOG%
) else (
    echo   STOPPED with code %RC%. Log: %LOG%
    echo   Re-run run_phase2.bat to resume.
)
echo ============================================================
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 40" 2>nul
echo.
echo Press any key to close.
pause >nul
exit /b %RC%


REM ====================================================================
:PHASE2
REM ====================================================================
echo ############################################################
echo # specbench Phase 2
echo # started: %DATE% %TIME%
echo ############################################################
echo.

%PY% -c "import sys; print('[env] python', sys.version)"
%PY% scripts\check_env.py
if errorlevel 1 (
    echo [FATAL] Environment check failed.
    exit /b 1
)
echo.

echo === STEP 1/5: EAGLE head structural tests ===
%PY% -m pytest tests\test_eagle_head.py -q
if errorlevel 1 (
    echo [FATAL] The head is not causal. Training it would produce a head that
    echo         reads its labels from the future and is useless at inference.
    exit /b 2
)
echo [STEP 1 OK]
echo.

echo === STEP 2/5: hidden states ===
%PY% scripts\run_phase2_collect.py
if errorlevel 1 (
    echo [FATAL] Hidden-state collection failed.
    exit /b 3
)
echo [STEP 2 OK]
echo.

echo === STEP 3/5: train EAGLE heads ===
%PY% scripts\run_phase2_train.py --epochs 3
if errorlevel 1 (
    echo [FATAL] Head training failed.
    exit /b 4
)
echo [STEP 3 OK]
echo.

echo === STEP 4/5: EAGLE benchmark sweep ===
%PY% scripts\run_phase2_sweep.py --k-values 1 2 3 4 6 --max-new-tokens 128
if errorlevel 1 (
    echo [FATAL] Sweep failed or was interrupted. Re-run to resume at the next head.
    exit /b 5
)
echo [STEP 4 OK]
echo.

echo === STEP 5/5: figures and results document ===
%PY% scripts\make_pareto.py
if errorlevel 1 echo [WARN] Figure generation failed.
%PY% scripts\make_results_doc.py
if errorlevel 1 echo [WARN] Could not regenerate docs\results.md.
echo [STEP 5 OK]
echo.

%PY% scripts\report_status.py
echo.
echo ############################################################
echo # finished: %DATE% %TIME%
echo ############################################################
exit /b 0
