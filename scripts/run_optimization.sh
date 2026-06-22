#!/bin/bash
# =============================================================================
# run_optimization.sh
# =============================================================================
# Unified SLURM launcher for hyperparameter and training optimization across
# all three genomic language models (DNABERT-2, HyenaDNA, Nucleotide Transformer v2).
#
# QUICK REFERENCE — edit the USER CONFIGURATION block below, then submit with:
#   sbatch run_optimization.sh
#
# Or override any variable on the command line:
#   MODEL=hyena OPT_TYPE=hp ALGORITHM=ts N_TRIALS=50 sbatch run_optimization.sh
#
# ── Models ────────────────────────────────────────────────────────────────────
#   dnabert2   zhihan1996/DNABERT-2-117M
#   hyena      LongSafari/hyenadna-tiny-16k-seqlen-d128-hf
#   nt         InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
#
# ── Optimization types ────────────────────────────────────────────────────────
#   hp     Hyperparameter search  (lr, wd, warmup, batch size, grad accum)
#   train  Training config search (epochs, scheduler, fp16, max_length, etc.)
#
# ── Algorithms ────────────────────────────────────────────────────────────────
#   gs     Grid Search
#   rs     Random Search
#   ts     Tree-structured Parzen Estimator / TPE  (Optuna)
#   ga     Genetic Algorithm
#   hc     Hill Climbing
#   sa     Simulated Annealing
#   sopt   Sequential GP-BO  (scikit-optimize)
#   bayes  Bayesian TPE       (Optuna)
# =============================================================================

# ── SLURM directives (edit time/mem to match your cluster) ───────────────────
#SBATCH --job-name=nlp_opt
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/opts/opt_%x_%j.out
#SBATCH --error=logs/opts/opt_%x_%j.err

set -euo pipefail
set -x

# =============================================================================
# USER CONFIGURATION  — change these values before submitting
# =============================================================================

# ── Which model to optimise: dnabert2 | hyena | nt ───────────────────────────
MODEL="${MODEL:-dnabert2}"

# ── Which search space: hp | train ───────────────────────────────────────────
OPT_TYPE="${OPT_TYPE:-hp}"

# ── Which algorithm: gs | rs | ts | ga | hc | sa | sopt | bayes ─────────────
ALGORITHM="${ALGORITHM:-ts}"

# ── Number of trials (function evaluations) ──────────────────────────────────
N_TRIALS="${N_TRIALS:-20}"

# ── Default number of training epochs used when not in the search space ──────
BASE_EPOCHS="${BASE_EPOCHS:-30}"

# ── Random seed ──────────────────────────────────────────────────────────────
SEED="${SEED:-42}"

# ── Window size for the CSV datasets: 100 | 200 | 500 ────────────────────────
WINDOW="${WINDOW:-100}"

# =============================================================================
# PATH CONFIGURATION  — adjust to your HPC layout
# =============================================================================

PROJECT_DIR=/home/hpc/users/ml_models/rebeka.maneva/nlp
CODE_DIR=${PROJECT_DIR}/code
DATA_DIR=${PROJECT_DIR}/data

# Pre-split DNABERT-2 data folder (produced by make_splits.py)
DNABERT2_DATA=${DATA_DIR}/dnabert2_${WINDOW}

# Raw CSVs for HyenaDNA / NT (split is done inside the python script)
HF_DATA_CSV=${DATA_DIR}/promoter_binary_${WINDOW}bp.csv

# DNABERT-2 official code directory (contains train.py)
DNABERT2_TRAIN_DIR=${CODE_DIR}/dnabert2_official

# Optimization python scripts (this script assumes they live in CODE_DIR)
OPT_HF_SCRIPT=${CODE_DIR}/optimize_hf.py
OPT_DNABERT2_SCRIPT=${CODE_DIR}/optimize_dnabert2.py

# Python packages
PY_PKGS_HF=${PROJECT_DIR}/py_pkgs
PY_PKGS_DNABERT2=${PROJECT_DIR}/py_pkgs_dnabert2

# train_hf_seqclf.py path (used by HyenaDNA / NT optimizer)
TRAIN_HF_SCRIPT=${CODE_DIR}/train_hf_seqclf.py

# Output root — a subdirectory is created per model/opt_type/algorithm/window
OUT_ROOT=${PROJECT_DIR}/outputs/opt/${MODEL}_${WINDOW}bp/${OPT_TYPE}/${ALGORITHM}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${OUT_ROOT}

export HF_HOME=/home/hpc/users/ml_models/rebeka.maneva/.cache/huggingface
export TRANSFORMERS_CACHE=/home/hpc/users/ml_models/rebeka.maneva/.cache/huggingface/transformers
export TOKENIZERS_PARALLELISM=false

# DNABERT-2 specific env (harmless to set for other models too)
export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1
export TRITON_INTERPRET=0
export CUDA_LAUNCH_BLOCKING=1
export FLASH_ATTENTION_DO_NOT_USE_TRITON=1
export TRITON_HOME=/home/hpc/users/ml_models/rebeka.maneva/.triton
export TRITON_CACHE_DIR=${TRITON_HOME}/cache
export TORCHINDUCTOR_CACHE_DIR=/home/hpc/users/ml_models/rebeka.maneva/.torchinductor
mkdir -p ${TRITON_HOME} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}

# =============================================================================
# DISPATCH
# =============================================================================

echo "============================================================"
echo "  Model     : ${MODEL}"
echo "  Opt type  : ${OPT_TYPE}"
echo "  Algorithm : ${ALGORITHM}"
echo "  Trials    : ${N_TRIALS}"
echo "  Window    : ${WINDOW} bp"
echo "  Output    : ${OUT_ROOT}"
echo "============================================================"

if [ "${MODEL}" = "dnabert2" ]; then
    # ── DNABERT-2 ─────────────────────────────────────────────────────────────
    export PYTHONPATH=${PY_PKGS_DNABERT2}

    python3 ${OPT_DNABERT2_SCRIPT} \
        --data-path   ${DNABERT2_DATA} \
        --train-dir   ${DNABERT2_TRAIN_DIR} \
        --output-dir  ${OUT_ROOT} \
        --py-pkgs     ${PY_PKGS_DNABERT2} \
        --opt-type    ${OPT_TYPE} \
        --algorithm   ${ALGORITHM} \
        --n-trials    ${N_TRIALS} \
        --base-epochs ${BASE_EPOCHS} \
        --seed        ${SEED}

elif [ "${MODEL}" = "hyena" ] || [ "${MODEL}" = "nt" ]; then
    # ── HyenaDNA / Nucleotide Transformer ─────────────────────────────────────
    export PYTHONPATH=${PY_PKGS_HF}:${PYTHONPATH:-}

    python3 ${OPT_HF_SCRIPT} \
        --model       ${MODEL} \
        --data-csv    ${HF_DATA_CSV} \
        --output-dir  ${OUT_ROOT} \
        --train-script ${TRAIN_HF_SCRIPT} \
        --opt-type    ${OPT_TYPE} \
        --algorithm   ${ALGORITHM} \
        --n-trials    ${N_TRIALS} \
        --base-epochs ${BASE_EPOCHS} \
        --seed        ${SEED}

else
    echo "ERROR: unknown MODEL '${MODEL}'. Must be: dnabert2 | hyena | nt"
    exit 1
fi

echo "============================================================"
echo "  Optimization complete. Results in: ${OUT_ROOT}"
echo "============================================================"
