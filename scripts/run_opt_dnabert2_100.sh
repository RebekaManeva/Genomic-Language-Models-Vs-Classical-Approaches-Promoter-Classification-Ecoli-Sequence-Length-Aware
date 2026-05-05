#!/bin/bash
#SBATCH --job-name=opt_dnabert2_100
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/opt_dnabert2_100_%j.out
#SBATCH --error=logs/opt_dnabert2_100_%j.err

# =============================================================================
# Bayesian optimization (Optuna) — DNABERT-2, 100 bp sequences
# =============================================================================

set -euo pipefail
set -x
BASE_DIR=${BASE_DIR:-$HOME/dnabert_project}

PROJECT_DIR=${BASE_DIR}
OPT_DIR=${PROJECT_DIR}/optimizations_nlp
DATA_PATH=${PROJECT_DIR}/data/dnabert2_100
OUT_DIR=${PROJECT_DIR}/outputs/opt_dnabert2_100bp
LOG_DIR=${PROJECT_DIR}/logs

mkdir -p ${LOG_DIR} ${OUT_DIR}


export PYTHONPATH=${PROJECT_DIR}/py_pkgs_dnabert2

export HF_HOME=${BASE_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers

export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export FLASH_ATTENTION_DO_NOT_USE_TRITON=1
export PYTHONNOUSERSITE=1

export TRITON_HOME=${BASE_DIR}/.triton
export TRITON_CACHE_DIR=${TRITON_HOME}/cache
export TORCHINDUCTOR_CACHE_DIR=${BASE_DIR}/.torchinductor

mkdir -p ${TRITON_HOME} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}

echo "Starting DNABERT-2 100bp optimization — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 ${OPT_DIR}/optimize_dnabert2.py \
  --data-path      ${DATA_PATH} \
  --model-name     zhihan1996/DNABERT-2-117M \
  --output-dir     ${OUT_DIR} \
  --n-trials       30 \
  --search-epochs  5 \
  --full-epochs    50 \
  --max-length     512 \
  --kmer           -1 \
  --fp16 \
  --study-name     dnabert2_100bp_opt \
  --storage        sqlite:///${OUT_DIR}/study.db

echo "Finished DNABERT-2 100bp optimization — $(date)"