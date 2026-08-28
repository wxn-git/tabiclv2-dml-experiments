@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHONPATH=%CD%\src"
set "MPLCONFIGDIR=%CD%\.matplotlib"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

if not exist "results\logs\stage2_parallel" mkdir "results\logs\stage2_parallel"

"C:\Users\Stern\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "scripts\run_stage2_parallel.py" --output-root "results\raw" --log-dir "results\logs\stage2_parallel" 1>> "results\logs\stage2_parallel\supervisor.stdout.log" 2>> "results\logs\stage2_parallel\supervisor.stderr.log"

set "EXIT_CODE=%ERRORLEVEL%"
echo %DATE% %TIME% EXITCODE %EXIT_CODE%>> "results\logs\stage2_parallel\supervisor.exit.log"
exit /b %EXIT_CODE%
