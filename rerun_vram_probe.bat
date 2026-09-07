@echo off
REM ====================================================================
REM  rerun_vram_probe.bat - re-run only the Phase 0 VRAM probe.
REM
REM  run_all.bat skips a completed stage, so this exists to re-measure the
REM  VRAM breakdown on its own after a change to the probe. It overwrites
REM  results\vram_probe.json and appends to a fresh log.
REM
REM  Safe to run while nothing else is using the GPU. Do not run it while
REM  run_all.bat is mid-benchmark: two processes competing for 4 GB of VRAM
REM  will either OOM or, worse, quietly distort the timings.
REM ====================================================================
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8:backslashreplace
set PYTHONUTF8=1

if not exist "logs" mkdir "logs"
set LOG=logs\vram_probe_rerun.log

echo Running VRAM probe. Output: %LOG%
.venv\Scripts\python.exe scripts\run_vram_probe.py > "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo.
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 30" 2>nul
echo.
if "%RC%"=="0" (echo VRAM probe OK.) else (echo VRAM probe exited with %RC%.)
echo Press any key to close.
pause >nul
exit /b %RC%
