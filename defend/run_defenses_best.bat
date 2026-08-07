@echo off
REM =============================================================================
REM run_defenses_best.bat — Run 4 Backdoor Defenses (STRIP, NoiSec, NC, Pruning)
REM =============================================================================
REM Evaluates stealthiness & defense resilience on trained backdoored model
REM (Best Setting: bend scenario, theta_max=20.0°, rho=0.3)
REM =============================================================================

setlocal enabledelayedexpansion

set CONFIG=%~dp0best_sweep_config.yaml
set CHECKPOINT=%~dp0..\experiments_out\victim_a\hpeli_bend_micro_dropper_s0\best.pt
set OUTDIR=%~dp0..\defend_out

:parse_loop
if "%~1"=="" goto done_parse
if /i "%~1"=="--checkpoint" ( set CHECKPOINT=%~2 & shift & shift & goto parse_loop )
if /i "%~1"=="--config"     ( set CONFIG=%~2     & shift & shift & goto parse_loop )
if /i "%~1"=="--outdir"     ( set OUTDIR=%~2     & shift & shift & goto parse_loop )
echo [ERROR] Unknown argument: %~1
exit /b 1
:done_parse

echo ============================================================
echo   Backdoor Defense Evaluation Suite
echo ============================================================
echo   Config     : %CONFIG%
echo   Checkpoint : %CHECKPOINT%
echo   Output     : %OUTDIR%
echo ============================================================
echo.
echo Running 4 Defense Methods:
echo   1. STRIP           (Runtime input perturbation entropy/dispersion)
echo   2. NoiSec          (Denoising Autoencoder reconstruction residual)
echo   3. Neural Cleanse  (MAD trigger reverse engineering index)
echo   4. Fine-Pruning    (Channel activation pruning & ASR post-repair)
echo.

cd /d "%~dp0.."
python -m defend.run_defenses --config "%CONFIG%" --checkpoint "%CHECKPOINT%" --outdir "%OUTDIR%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Defense evaluation failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   Defense Evaluation Completed.
echo   Summary JSON : %OUTDIR%\summary.json
echo   Plot Figure  : %OUTDIR%\defenses_summary.png
echo ============================================================
endlocal
