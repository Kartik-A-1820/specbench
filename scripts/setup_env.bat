@echo off
REM ====================================================================
REM  scripts\setup_env.bat - build an isolated environment for specbench.
REM
REM  Creates .venv next to the project and installs a CUDA build of torch
REM  plus the rest of requirements.txt into it.
REM
REM  Why a virtual environment rather than the system interpreter: the
REM  system Python here already has torch 2.14.0+cpu installed. Replacing
REM  it globally would silently break whatever else depends on it. A venv
REM  keeps the CUDA build local to this project and leaves the rest of the
REM  machine untouched.
REM
REM  Safe to re-run. Skips creation if .venv already exists, and pip skips
REM  packages that are already satisfied.
REM ====================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

echo [setup_env] Project: %CD%
echo.

echo [setup_env] --- GPU and driver ---
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [setup_env] nvidia-smi not found on PATH. Cannot confirm the driver.
) else (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
)
echo.

REM --- Locate a base interpreter to build the venv from ---------------
set BASEPY=
where py >nul 2>&1 && set BASEPY=py -3.11
if "%BASEPY%"=="" (
    where python >nul 2>&1 && set BASEPY=python
)
if "%BASEPY%"=="" (
    echo [setup_env] FATAL: no python found on PATH.
    exit /b 1
)
echo [setup_env] Base interpreter: %BASEPY%

if exist ".venv\Scripts\python.exe" (
    echo [setup_env] .venv already exists.
) else (
    echo [setup_env] Creating .venv ...
    %BASEPY% -m venv .venv
    if errorlevel 1 (
        echo [setup_env] `py -3.11 -m venv` failed; retrying with plain python.
        python -m venv .venv
        if errorlevel 1 (
            echo [setup_env] FATAL: could not create a virtual environment.
            exit /b 1
        )
    )
)

set VPY=.venv\Scripts\python.exe
%VPY% -c "import sys; print('[setup_env] venv python', sys.version)"

echo.
echo [setup_env] Upgrading pip ...
%VPY% -m pip install --disable-pip-version-check -q --upgrade pip setuptools wheel
if errorlevel 1 echo [setup_env] WARNING: pip upgrade failed; continuing.

echo.
echo [setup_env] Installing torch 2.5.1 + CUDA 12.1 (about 2.5 GB, this takes a while) ...
%VPY% -m pip install --disable-pip-version-check torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo [setup_env] cu121 wheel failed. Trying cu118, which supports older drivers.
    %VPY% -m pip install --disable-pip-version-check torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [setup_env] FATAL: could not install a CUDA build of torch.
        echo               Check network access to download.pytorch.org.
        exit /b 2
    )
)

echo.
echo [setup_env] Installing the remaining requirements ...
%VPY% -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [setup_env] FATAL: requirements install failed.
    exit /b 3
)

echo.
echo [setup_env] Verifying ...
%VPY% scripts\check_env.py
if errorlevel 1 (
    echo [setup_env] FATAL: the environment still does not pass its checks.
    exit /b 4
)

echo.
echo [setup_env] Environment ready at .venv
exit /b 0
