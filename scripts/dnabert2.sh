#!/bin/bash
#SBATCH --job-name=dnabert2_500
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dnabert2_500_%j.out
#SBATCH --error=logs/dnabert2_500_%j.err
 
set -euo pipefail
set -x
 
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_PATH=${PROJECT_DIR}/data/dnabert2_500
OUT_DIR=${PROJECT_DIR}/outputs/dnabert2_500bp
OFFICIAL_DIR=${PROJECT_DIR}/code/dnabert2_official
 
mkdir -p logs
mkdir -p ${OUT_DIR}
 
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1
export TRITON_INTERPRET=0
export CUDA_LAUNCH_BLOCKING=1
export FLASH_ATTENTION_DO_NOT_USE_TRITON=1
 
export TRITON_HOME=${HOME}/.triton
export TRITON_CACHE_DIR=${HOME}/.triton/cache
export TORCHINDUCTOR_CACHE_DIR=${HOME}/.torchinductor
mkdir -p ${TRITON_HOME}
mkdir -p ${TRITON_CACHE_DIR}
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}
 
cd ${OFFICIAL_DIR}
 
python3 ${PROJECT_DIR}/code/train_dnabert.py \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --data_path ${DATA_PATH} \
  --kmer -1 \
  --run_name dnabert2_promoter_500 \
  --model_max_length 25 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 200 \
  --output_dir ${OUT_DIR} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 100 \
  --overwrite_output_dir True \
  --log_level info \
  --find_unused_parameters False