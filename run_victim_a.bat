@echo off
REM =============================================================================
REM run_victim_a.bat  —  Victim Branch A: 27 runs
REM   3 models  x  3 scenarios  x  1 trigger (MicroDoppler)  x  3 seeds
REM   = 27 runs  →  mean±std per (model, scenario)
REM =============================================================================
REM Usage:
REM   run_victim_a.bat                       <- sequential, default output dir
REM   run_victim_a.bat --parallel 9          <- 9 concurrent runs (all seeds)
REM   run_victim_a.bat --parallel 4 --gpu 0  <- 4 concurrent on GPU 0
REM   run_victim_a.bat --epochs 2            <- quick smoke test (2 epochs)
REM   run_victim_a.bat --seeds 0             <- single seed (9 runs only)
REM   run_victim_a.bat --outdir my_out_dir   <- custom output directory
REM =============================================================================

setlocal enabledelayedexpansion

REM --- Defaults ---
set GPU=
set GPUS=
set EPOCHS=
set PARALLEL=1
set SEEDS=0 1 2
set OUTDIR=%~dp0experiments_out\victim_a

REM --- Parse args ---
:parse_loop
if "%~1"=="" goto done_parse
if /i "%~1"=="--gpu"       ( set GPU=%~2       & shift & shift & goto parse_loop )
if /i "%~1"=="--gpus"      ( set GPUS=%~2      & shift & shift & goto parse_loop )
if /i "%~1"=="--epochs"    ( set EPOCHS=%~2    & shift & shift & goto parse_loop )
if /i "%~1"=="--parallel"  ( set PARALLEL=%~2  & shift & shift & goto parse_loop )
if /i "%~1"=="--seeds"     ( set SEEDS=%~2     & shift & shift & goto parse_loop )
if /i "%~1"=="--outdir"    ( set OUTDIR=%~2    & shift & shift & goto parse_loop )
echo [ERROR] Unknown argument: %~1
exit /b 1
:done_parse

REM --- Ensure output directory exists ---
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM --- Build python args ---
set PYTHON_ARGS=--dataset mmfi
set PYTHON_ARGS=%PYTHON_ARGS% --models hpeli metafiplusplus graphposefi
set PYTHON_ARGS=%PYTHON_ARGS% --scenarios bend cross nod
set PYTHON_ARGS=%PYTHON_ARGS% --triggers micro_dropper
set PYTHON_ARGS=%PYTHON_ARGS% --seeds %SEEDS%
set PYTHON_ARGS=%PYTHON_ARGS% --outdir "%OUTDIR%"
set PYTHON_ARGS=%PYTHON_ARGS% --parallel %PARALLEL%

REM --- Device ---
if not "%GPUS%"==""  ( set PYTHON_ARGS=%PYTHON_ARGS% --gpus %GPUS% ) else (
if not "%GPU%"==""   ( set PYTHON_ARGS=%PYTHON_ARGS% --device cuda:%GPU% ) )

if not "%EPOCHS%"=="" set PYTHON_ARGS=%PYTHON_ARGS% --epochs %EPOCHS%

REM --- Print plan ---
echo.
echo ============================================================
echo   Victim Branch A ^| MMFI Dataset ^| 27 Runs
echo ============================================================
echo   Models    : hpeli   metafiplusplus   graphposefi
echo   Scenarios : bend    cross            nod
echo   Trigger   : micro_dropper (MicroDoppler)
echo   Seeds     : %SEEDS%
echo   Total     : 3 x 3 x 1 x 3 = 27 runs
echo   Output    : %OUTDIR%
if not "%GPU%"==""  echo   GPU       : cuda:%GPU%
if not "%GPUS%"=="" echo   GPUs      : %GPUS%
echo   Parallel  : %PARALLEL% workers
if not "%EPOCHS%"=="" echo   Epochs    : %EPOCHS%
echo ============================================================
echo.
echo Metrics logged per run:
echo   Clean: MPJPE(down)  PA-MPJPE(down)  PCK@0.5(up)
echo   Attack: ASR(up)  Landed(up)  Preserved(up)  Plausible(up)  T-MPJPE(down)
echo   Quality: Spearman-rho(up)
echo.

REM --- Run ---
cd /d "%~dp0"
echo python run_experiments.py %PYTHON_ARGS%
echo.
python run_experiments.py %PYTHON_ARGS%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] run_experiments.py exited with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   All 27 runs completed.
echo ============================================================
echo   Per-seed results  : %OUTDIR%\results.csv
echo   Aggregated mean+-std : %OUTDIR%\results_agg.csv
echo ============================================================
endlocal
