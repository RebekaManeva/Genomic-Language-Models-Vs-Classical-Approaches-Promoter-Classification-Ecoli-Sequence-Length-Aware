#!/bin/bash
#SBATCH --job-name=prom_nt_100
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nt_%j.out
#SBATCH --error=logs/nt_%j.err

set -euo pipefail
set -x

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_CSV=${PROJECT_DIR}/data/promoter_binary_100bp.csv
OUT_DIR=${PROJECT_DIR}/outputs/nt_100bp

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${OUT_DIR}

export PYTHONPATH=${PROJECT_DIR}/py_pkgs:${PYTHONPATH:-}
export HF_HOME=${PROJECT_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${PROJECT_DIR}/.cache/huggingface/transformers

python3 ${PROJECT_DIR}/code/train_hf_seqclf.py \
  --train-csv ${DATA_CSV} \
  --split-from-single-csv \
  --model-name InstaDeepAI/nucleotide-transformer-v2-100m-multi-species \
  --output-dir ${OUT_DIR} \
  --max-length 512 \
  --epochs 30 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --gradient-accumulation-steps 2 \
  --fp16 \
  --gradient-checkpointing \
  --trust-remote-code
