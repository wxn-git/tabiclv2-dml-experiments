@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHONPATH=%CD%\src"
set "MPLCONFIGDIR=%CD%\.matplotlib"
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"

if not exist "results\logs" mkdir "results\logs"

"C:\Users\Stern\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "scripts\run_stage1.py" --output-root "results\raw" 1>> "results\logs\stage1_fresh_scheduled.stdout.log" 2>> "results\logs\stage1_fresh_scheduled.stderr.log"

echo %DATE% %TIME% EXITCODE %ERRORLEVEL%>> "results\logs\stage1_fresh_scheduled.exit.log"

