@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHONPATH=%CD%\src"
set "MPLCONFIGDIR=%CD%\.matplotlib"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"

if not exist "results\logs\stage2_resume_8" mkdir "results\logs\stage2_resume_8"

"C:\Users\Stern\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "scripts\run_stage2_resume_parallel.py" --cpu-workers 8 --output-root "results\raw" --log-dir "results\logs\stage2_resume_8" 1>> "results\logs\stage2_resume_8\supervisor.stdout.log" 2>> "results\logs\stage2_resume_8\supervisor.stderr.log"

set "EXIT_CODE=%ERRORLEVEL%"
echo %DATE% %TIME% EXITCODE %EXIT_CODE%>> "results\logs\stage2_resume_8\supervisor.exit.log"
exit /b %EXIT_CODE%
