#!/bin/bash
#SBATCH --job-name=prom_hyena_200
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hyena_%j.out
#SBATCH --error=logs/hyena_%j.err

set -euo pipefail
set -x

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_CSV=${PROJECT_DIR}/data/promoter_binary_200bp.csv
OUT_DIR=${PROJECT_DIR}/outputs/hyena_200bp

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${OUT_DIR}

export PYTHONPATH=${PROJECT_DIR}/py_pkgs:${PYTHONPATH:-}
export HF_HOME=${PROJECT_DIR}/.cache/huggingface
export TRANSFORMERS_CACHE=${PROJECT_DIR}/.cache/huggingface/transformers

python3 ${PROJECT_DIR}/code/train_hf_seqclf.py \
  --train-csv ${DATA_CSV} \
  --split-from-single-csv \
  --model-name LongSafari/hyenadna-tiny-16k-seqlen-d128-hf \
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
