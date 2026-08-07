#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh  —  MMFI backdoor experiment runner
# =============================================================================
# Mặc định: 1 scenario (ln) × 3 models × 3 triggers = 9 runs trên MMFI
#
# Parallel execution (Blackwell GPU)
# -----------------------------------
#   --parallel N     : N worker processes trên cùng 1 GPU (shared VRAM)
#   --gpus "0 1"     : round-robin sang nhiều GPU (N processes chia đều)
#
#   Khuyến nghị theo GPU:
#     RTX 5090  (32 GB)  → --parallel 9  (tất cả 9 runs song song)
#     RTX 5080  (16 GB)  → --parallel 4
#     B200      (96 GB)  → --parallel 9
#     GB200 NVL (192 GB) → --parallel 9  (thêm --gpus "0 1" nếu có 2 GPU)
#
# Cách dùng
# ---------
#   bash run_all_experiments.sh                          # sequential
#   bash run_all_experiments.sh --parallel 9             # Blackwell full parallel
#   bash run_all_experiments.sh --parallel 9 --gpu 0
#   bash run_all_experiments.sh --parallel 9 --gpus "0 1"
#   bash run_all_experiments.sh --scenario sqrt          # đổi scenario
#   bash run_all_experiments.sh --parallel 9 --epochs 5 --fast  # quick test
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${SCRIPT_DIR}/experiments_out"
LOG_FILE="${OUTDIR}/run_all.log"
GPU=""
GPUS=""
EPOCHS=""
MODELS="hpeli metafiplusplus graphposefi"
SCENARIO="ln"
TRIGGERS="micro_dropper wanet sig blended"
DATASET="mmfi"
PYTHON_BIN=""
FAST=""
PARALLEL=1      # default sequential; set --parallel 9 for Blackwell

# --------------------------------------------------------------------------- #
# Parse args
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";        shift 2 ;;
        --gpus)      GPUS="$2";       shift 2 ;;
        --epochs)    EPOCHS="$2";     shift 2 ;;
        --models)    MODELS="$2";     shift 2 ;;
        --scenario)  SCENARIO="$2";   shift 2 ;;
        --scenarios) SCENARIO="$2";   shift 2 ;;
        --triggers)  TRIGGERS="$2";   shift 2 ;;
        --dataset)   DATASET="$2";    shift 2 ;;
        --outdir)    OUTDIR="$2";     shift 2 ;;
        --python)    PYTHON_BIN="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2";   shift 2 ;;
        --fast)      FAST="1";        shift 1 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Auto-detect Python interpreter
# --------------------------------------------------------------------------- #
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in \
        "python" "python3" \
        "${HOME}/miniconda3/bin/python3" \
        "${HOME}/miniconda/bin/python3" \
        "${HOME}/anaconda3/bin/python3" \
        "/opt/conda/bin/python3" \
        "/usr/bin/python3"; do
        if command -v "${candidate}" &>/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Cannot find a Python interpreter."
    echo "  → Activate your conda/venv environment first, or pass --python /path/to/python"
    exit 1
fi

echo "[info] Using Python: ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
mkdir -p "${OUTDIR}"
LOG_FILE="${OUTDIR}/run_all.log"

# Build python command args array
PYTHON_ARGS=(
    "--models"    ${MODELS}
    "--scenarios" ${SCENARIO}
    "--triggers"  ${TRIGGERS}
    "--dataset"   "${DATASET}"
    "--outdir"    "${OUTDIR}"
    "--parallel"  "${PARALLEL}"
)

# Device selection: --gpus takes priority over --gpu
if [[ -n "${GPUS}" ]]; then
    # Multi-GPU: pass as --gpus 0 1 2 ...
    PYTHON_ARGS+=("--gpus" ${GPUS})
elif [[ -n "${GPU}" ]]; then
    PYTHON_ARGS+=("--device" "cuda:${GPU}")
fi

[[ -n "${EPOCHS}" ]] && PYTHON_ARGS+=("--epochs" "${EPOCHS}")
[[ -n "${FAST}" ]]   && PYTHON_ARGS+=("--poison-select" "uniform")

# --------------------------------------------------------------------------- #
# Print plan
# --------------------------------------------------------------------------- #
N_MODELS=$(echo ${MODELS}   | wc -w)
N_SCEN=$(echo   ${SCENARIO} | wc -w)
N_TRIG=$(echo   ${TRIGGERS} | wc -w)
TOTAL=$(( N_MODELS * N_SCEN * N_TRIG ))

echo "============================================================"
echo " Backdoor Experiment Suite — MMFI (Blackwell-ready)"
echo "============================================================"
echo " Script dir : ${SCRIPT_DIR}"
echo " Output dir : ${OUTDIR}"
echo " Dataset    : ${DATASET}"
echo " GPU(s)     : ${GPUS:-${GPU:-auto}}"
echo " Parallel   : ${PARALLEL} workers"
echo " Epochs     : ${EPOCHS:-from config}"
echo " Models     : ${MODELS}"
echo " Scenarios  : ${SCENARIO}"
echo " Triggers   : ${TRIGGERS}"
echo ""
echo " Total runs : ${TOTAL}  (${N_MODELS} models × ${N_SCEN} scenario × ${N_TRIG} triggers)"
if [[ "${PARALLEL}" -ge "${TOTAL}" ]]; then
    echo " Mode       : FULL PARALLEL (all ${TOTAL} runs simultaneously)"
elif [[ "${PARALLEL}" -gt 1 ]]; then
    echo " Mode       : PARALLEL (${PARALLEL} workers, $(( (TOTAL + PARALLEL - 1) / PARALLEL )) waves)"
else
    echo " Mode       : SEQUENTIAL"
fi
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
cd "${SCRIPT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${TOTAL} runs..." | tee -a "${LOG_FILE}"
echo "Command: ${PYTHON_BIN} run_experiments.py ${PYTHON_ARGS[*]}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" run_experiments.py "${PYTHON_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All ${TOTAL} runs completed." | tee -a "${LOG_FILE}"
    echo ""
    echo "Results → ${OUTDIR}/results.csv"
    echo "Log     → ${LOG_FILE}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: exited with code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
    exit ${EXIT_CODE}
fi
