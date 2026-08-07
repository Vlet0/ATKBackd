@echo off
REM =============================================================================
REM run_all_experiments.bat  —  MMFI backdoor runner (Windows, Blackwell-ready)
REM =============================================================================
REM Mặc định: scenario ln x 3 models x 3 triggers = 9 runs
REM
REM Blackwell parallel:
REM   run_all_experiments.bat --parallel 9           <- tất cả 9 runs song song
REM   run_all_experiments.bat --parallel 9 --gpu 0
REM   run_all_experiments.bat --parallel 4 --epochs 5  <- quick test
REM   run_all_experiments.bat --parallel 9 --gpus "0 1"  <- 2 GPU
REM =============================================================================

setlocal enabledelayedexpansion

REM --- Defaults ---
set SCENARIO=ln
set MODELS=hpeli metafiplusplus graphposefi
set TRIGGERS=micro_dropper wanet sig blended
set DATASET=mmfi
set GPU=
set GPUS=
set EPOCHS=
set PARALLEL=1
set FAST=
set OUTDIR=%~dp0experiments_out

REM --- Parse args ---
:parse_loop
if "%~1"=="" goto done_parse
if /i "%~1"=="--gpu"       ( set GPU=%~2       & shift & shift & goto parse_loop )
if /i "%~1"=="--gpus"      ( set GPUS=%~2      & shift & shift & goto parse_loop )
if /i "%~1"=="--epochs"    ( set EPOCHS=%~2    & shift & shift & goto parse_loop )
if /i "%~1"=="--scenario"  ( set SCENARIO=%~2  & shift & shift & goto parse_loop )
if /i "%~1"=="--scenarios" ( set SCENARIO=%~2  & shift & shift & goto parse_loop )
if /i "%~1"=="--models"    ( set MODELS=%~2    & shift & shift & goto parse_loop )
if /i "%~1"=="--triggers"  ( set TRIGGERS=%~2  & shift & shift & goto parse_loop )
if /i "%~1"=="--parallel"  ( set PARALLEL=%~2  & shift & shift & goto parse_loop )
if /i "%~1"=="--outdir"    ( set OUTDIR=%~2    & shift & shift & goto parse_loop )
if /i "%~1"=="--fast"      ( set FAST=1        & shift         & goto parse_loop )
echo [ERROR] Unknown argument: %~1
exit /b 1
:done_parse

REM --- Ensure output directory exists ---
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM --- Build python args ---
set PYTHON_ARGS=--models %MODELS% --scenarios %SCENARIO% --triggers %TRIGGERS% --dataset %DATASET% --outdir "%OUTDIR%" --parallel %PARALLEL%

REM Device: --gpus takes priority over --gpu
if not "%GPUS%"==""   ( set PYTHON_ARGS=%PYTHON_ARGS% --gpus %GPUS% ) else (
if not "%GPU%"==""    ( set PYTHON_ARGS=%PYTHON_ARGS% --device cuda:%GPU% ) )

if not "%EPOCHS%"=="" set PYTHON_ARGS=%PYTHON_ARGS% --epochs %EPOCHS%
if not "%FAST%"==""   set PYTHON_ARGS=%PYTHON_ARGS% --poison-select uniform

REM --- Print plan ---
echo ============================================================
echo  Backdoor Experiment Suite - MMFI (Blackwell-ready)
echo ============================================================
echo  Output dir : %OUTDIR%
echo  Dataset    : %DATASET%
echo  GPU(s)     : %GPUS%%GPU%
echo  Parallel   : %PARALLEL% workers
echo  Epochs     : %EPOCHS%
echo  Models     : %MODELS%
echo  Scenarios  : %SCENARIO%
echo  Triggers   : %TRIGGERS%
echo ============================================================
echo.

REM --- Run ---
cd /d "%~dp0"
echo python run_experiments.py %PYTHON_ARGS%
python run_experiments.py %PYTHON_ARGS%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] run_experiments.py exited with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo All runs completed.
echo Results: %OUTDIR%\results.csv
endlocal
