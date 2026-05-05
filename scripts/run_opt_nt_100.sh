#!/bin/bash
#SBATCH --job-name=opt_nt_100
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hpc/users/ml_models/rebeka.maneva/nlp/logs/opt_nt_100_%j.out
#SBATCH --error=/home/hpc/users/ml_models/rebeka.maneva/nlp/logs/opt_nt_100_%j.err

# =============================================================================
# Bayesian optimization (Optuna) — Nucleotide Transformer v2 100M, 100 bp
# =============================================================================

set -euo pipefail
set -x

BASE_DIR=${BASE_DIR:-$HOME/dnabert_project}

PROJECT_DIR=${BASE_DIR}
OPT_DIR=${PROJECT_DIR}/optimizations_nlp
DATA_CSV=${PROJECT_DIR}/data/promoter_binary_100bp.csv
OUT_DIR=${PROJECT_DIR}/outputs/opt_nt_100bp
LOG_DIR=${PROJECT_DIR}/logs

mkdir -p ${LOG_DIR} ${OUT_DIR}

export PYTHONPATH=${PROJECT_DIR}/py_pkgs

export HF_HOME=${BASE_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers

export TOKENIZERS_PARALLELISM=false

echo "Starting NT 100bp optimization — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 ${OPT_DIR}/optimize_hf_seqclf.py \
  --train-csv           ${DATA_CSV} \
  --split-from-single-csv \
  --model-name          InstaDeepAI/nucleotide-transformer-v2-100m-multi-species \
  --output-dir          ${OUT_DIR} \
  --max-length          512 \
  --n-trials            30 \
  --search-epochs       5 \
  --full-epochs         50 \
  --fp16 \
  --gradient-checkpointing \
  --trust-remote-code \
  --study-name          nt_100bp_opt \
  --storage             sqlite:///${OUT_DIR}/study.db

echo "Finished NT 100bp optimization — $(date)"
