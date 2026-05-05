#!/bin/bash
#SBATCH --job-name=opt_hyena_200
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hpc/users/ml_models/rebeka.maneva/nlp/logs/opt_hyena_200_%j.out
#SBATCH --error=/home/hpc/users/ml_models/rebeka.maneva/nlp/logs/opt_hyena_200_%j.err

# =============================================================================
# Bayesian optimization (Optuna) — HyenaDNA tiny-16k-d128, 200 bp sequences
# =============================================================================

set -euo pipefail
set -x

BASE_DIR=${BASE_DIR:-$HOME/dnabert_project}

PROJECT_DIR=${BASE_DIR}
OPT_DIR=${PROJECT_DIR}/optimizations_nlp
DATA_CSV=${PROJECT_DIR}/data/promoter_binary_100bp.csv
OUT_DIR=${PROJECT_DIR}/outputs/opt_hyena_100bp
LOG_DIR=${PROJECT_DIR}/logs

mkdir -p ${LOG_DIR} ${OUT_DIR}

export PYTHONPATH=${PROJECT_DIR}/py_pkgs

export HF_HOME=${BASE_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers

export TOKENIZERS_PARALLELISM=false

echo "Starting HyenaDNA 200bp optimization — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 ${OPT_DIR}/optimize_hf_seqclf.py \
  --train-csv           ${DATA_CSV} \
  --split-from-single-csv \
  --model-name          LongSafari/hyenadna-tiny-16k-seqlen-d128-hf \
  --output-dir          ${OUT_DIR} \
  --max-length          512 \
  --n-trials            30 \
  --search-epochs       5 \
  --full-epochs         50 \
  --fp16 \
  --gradient-checkpointing \
  --trust-remote-code \
  --study-name          hyena_200bp_opt \
  --storage             sqlite:///${OUT_DIR}/study.db

echo "Finished HyenaDNA 200bp optimization — $(date)"
