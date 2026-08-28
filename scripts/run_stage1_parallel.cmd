@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHONPATH=%CD%\src"
set "MPLCONFIGDIR=%CD%\.matplotlib"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

if not exist "results\logs\stage1_parallel" mkdir "results\logs\stage1_parallel"

"C:\Users\Stern\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "scripts\run_stage1_parallel.py" --cpu-workers 12 --output-root "results\raw" --log-dir "results\logs\stage1_parallel" 1>> "results\logs\stage1_parallel\supervisor.stdout.log" 2>> "results\logs\stage1_parallel\supervisor.stderr.log"

echo %DATE% %TIME% EXITCODE %ERRORLEVEL%>> "results\logs\stage1_parallel\supervisor.exit.log"
