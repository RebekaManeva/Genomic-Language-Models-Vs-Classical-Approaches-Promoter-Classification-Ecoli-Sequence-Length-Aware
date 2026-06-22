#!/bin/bash
# =============================================================================
# smoke_test_all.sh
# =============================================================================
# Smoke test for optimize_hf_seqclf.py and optimize_dnabert2.py.
# Runs every algorithm with 1 trial / 1 epoch to verify pipelines end-to-end.
#
# Submit:  sbatch smoke_test_all.sh
# Local:   bash smoke_test_all.sh
# =============================================================================

#SBATCH --job-name=smoke_test
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

set -euo pipefail
set -x

# =============================================================================
# PATHS
# =============================================================================

PROJECT_DIR=/home/hpc/users/ml_models/rebeka.maneva/nlp
OPT_DIR=${PROJECT_DIR}/optimizations
DATA_CSV=${PROJECT_DIR}/data/promoter_binary_500bp.csv
DNABERT2_DATA=${PROJECT_DIR}/data/dnabert2_500
SMOKE_OUT=${PROJECT_DIR}/outputs/smoke_test
LOGS=${PROJECT_DIR}/logs

# =============================================================================
# VENV INTERPRETERS
#
# Each model family has its own venv with the correct torch/tokenizers/
# transformers already installed and verified. We invoke these directly
# via env -i, which means:
#   - No PYTHONPATH needed — the venv's site-packages are on sys.path
#     automatically via the venv's pyvenv.cfg
#   - No sitecustomize.py hacks needed
#   - No tokenizers version bleed between environments
# =============================================================================

PYTHON_HYENA=/home/hpc/users/ml_models/rebeka.maneva/venv_hyenadna/bin/python
PYTHON_NT=/home/hpc/users/ml_models/rebeka.maneva/venv_nt/bin/python
PYTHON_DB2=/home/hpc/users/ml_models/rebeka.maneva/venv_dnabert2/bin/python

# Validate all interpreters exist before starting
for _label_py in "venv_hyenadna:${PYTHON_HYENA}" "venv_nt:${PYTHON_NT}" "venv_dnabert2:${PYTHON_DB2}"; do
    _label="${_label_py%%:*}"
    _py="${_label_py##*:}"
    if [ ! -x "${_py}" ]; then
        echo "ERROR: interpreter not found or not executable: ${_py} (${_label})"
        exit 1
    fi
done

mkdir -p ${LOGS} ${SMOKE_OUT}

# =============================================================================
# SHARED ENV VARS
# =============================================================================

HF_HOME=/home/hpc/users/ml_models/rebeka.maneva/.cache/huggingface
TRITON_HOME=/home/hpc/users/ml_models/rebeka.maneva/.triton
TRITON_CACHE_DIR=${TRITON_HOME}/cache
TORCHINDUCTOR_CACHE_DIR=/home/hpc/users/ml_models/rebeka.maneva/.torchinductor

mkdir -p ${TRITON_HOME} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}

# =============================================================================
# SMOKE TEST CONFIG
# =============================================================================

N_TRIALS=1
SEARCH_EPOCHS=1
FULL_EPOCHS=1
SEED=42

ALGORITHMS=(rs ts bayes ga hc sa)

# Each entry: "model_key:model_id:interpreter_path"
HF_MODELS=(
    "hyena:LongSafari/hyenadna-tiny-16k-seqlen-d128-hf:${PYTHON_HYENA}"
    "nt:InstaDeepAI/nucleotide-transformer-v2-100m-multi-species:${PYTHON_NT}"
)

# =============================================================================
# HEADER
# =============================================================================

echo "======================================================================"
echo "SMOKE TEST — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A (login node)')"
echo ""
echo "Interpreters:"
for _label_py in "hyena:${PYTHON_HYENA}" "nt:${PYTHON_NT}" "dnabert2:${PYTHON_DB2}"; do
    _label="${_label_py%%:*}"
    _py="${_label_py##*:}"
    _tok=$(${_py} -c 'import tokenizers; print(tokenizers.__version__)' 2>/dev/null || echo 'ERR')
    _tr=$(${_py}  -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'ERR')
    _th=$(${_py}  -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'ERR')
    echo "  ${_label}: torch=${_th}  transformers=${_tr}  tokenizers=${_tok}"
done
echo ""
echo "Algorithms : ${ALGORITHMS[*]}"
echo "Trials     : ${N_TRIALS} x ${SEARCH_EPOCHS} epoch(s) + ${FULL_EPOCHS} final"
echo "======================================================================"

PASS=0
FAIL=0
ERRORS=()

# =============================================================================
# HELPER
#
# run_test <name> <out_dir> <interpreter> <script> [args...]
#
# Runs <interpreter> inside a clean env -i environment.
# The venv interpreter resolves its own site-packages from pyvenv.cfg,
# so no PYTHONPATH is needed and no package bleed is possible.
# =============================================================================

run_test() {
    local name="$1";        shift
    local out_dir="$1";     shift
    local interpreter="$1"; shift

    echo ""
    echo ">>> TEST: ${name}"
    mkdir -p "${out_dir}"

    if env -i \
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        HOME="${HOME}" \
        HF_HOME="${HF_HOME}" \
        TRANSFORMERS_CACHE="${HF_HOME}/transformers" \
        TOKENIZERS_PARALLELISM="false" \
        TORCHDYNAMO_DISABLE="1" \
        CUDA_LAUNCH_BLOCKING="1" \
        FLASH_ATTENTION_DO_NOT_USE_TRITON="1" \
        TRITON_HOME="${TRITON_HOME}" \
        TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
        TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "${interpreter}" "$@" 2>&1 | tee "${out_dir}/smoke.log"; then
        echo "    [PASS] ${name}"
        PASS=$((PASS + 1))
    else
        echo "    [FAIL] ${name}  — see ${out_dir}/smoke.log"
        FAIL=$((FAIL + 1))
        ERRORS+=("${name}")
        echo "    --- tail of log ---"
        tail -n 20 "${out_dir}/smoke.log" | sed 's/^/    | /'
        echo "    -------------------"
    fi
}

# =============================================================================
# HF MODELS (HyenaDNA + Nucleotide Transformer)
# =============================================================================

for model_entry in "${HF_MODELS[@]}"; do
    IFS=':' read -r model_key model_id interpreter <<< "${model_entry}"

    for algo in "${ALGORITHMS[@]}"; do
        name="${model_key}_${algo}"
        out_dir="${SMOKE_OUT}/hf/${model_key}/${algo}"

        run_test "${name}" "${out_dir}" "${interpreter}" \
            ${OPT_DIR}/optimize_hf_seqclf.py \
                --train-csv             ${DATA_CSV} \
                --split-from-single-csv \
                --model-name            ${model_id} \
                --output-dir            ${out_dir} \
                --algorithm             ${algo} \
                --n-trials              ${N_TRIALS} \
                --search-epochs         ${SEARCH_EPOCHS} \
                --full-epochs           ${FULL_EPOCHS} \
                --seed                  ${SEED} \
                --trust-remote-code \
                --study-name            smoke_${model_key}_${algo} \
                --storage               sqlite:///${out_dir}/study.db
    done
done

# =============================================================================
# DNABERT-2
# =============================================================================

for algo in "${ALGORITHMS[@]}"; do
    name="dnabert2_${algo}"
    out_dir="${SMOKE_OUT}/dnabert2/${algo}"

    run_test "${name}" "${out_dir}" "${PYTHON_DB2}" \
        ${OPT_DIR}/optimize_dnabert2.py \
            --data-path     ${DNABERT2_DATA} \
            --output-dir    ${out_dir} \
            --algorithm     ${algo} \
            --n-trials      ${N_TRIALS} \
            --search-epochs ${SEARCH_EPOCHS} \
            --full-epochs   ${FULL_EPOCHS} \
            --seed          ${SEED} \
            --study-name    smoke_dnabert2_${algo} \
            --storage       sqlite:///${out_dir}/study.db
done

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL=$((PASS + FAIL))

echo ""
echo "======================================================================"
echo "SMOKE TEST SUMMARY — $(date)"
echo "  PASSED : ${PASS} / ${TOTAL}"
echo "  FAILED : ${FAIL} / ${TOTAL}"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "  FAILED TESTS:"
    for e in "${ERRORS[@]}"; do echo "    - ${e}"; done
    echo "  Logs: ${SMOKE_OUT}/<model>/<algo>/smoke.log"
    echo "======================================================================"
    exit 1
fi

echo "  All pipelines and algorithms operational."
echo "======================================================================"
