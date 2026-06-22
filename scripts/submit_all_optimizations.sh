#!/bin/bash
# =============================================================================
# submit_all_optimizations.sh
# =============================================================================
# Submits one SLURM job per (model × opt_type × algorithm) combination.
# Edit the arrays below to control which combinations are submitted.
#
# Usage:
#   bash submit_all_optimizations.sh
#
# Dry-run (print commands without submitting):
#   DRY_RUN=1 bash submit_all_optimizations.sh
# =============================================================================

# =============================================================================
# CONFIGURATION — edit these
# =============================================================================

# Which models to run (space-separated subset of: dnabert2 hyena nt)
MODELS=( dnabert2 hyena nt )

# Which opt types (space-separated subset of: hp train)
OPT_TYPES=( hp train )

# Which algorithms (space-separated subset of: gs rs ts ga hc sa bayes)
ALGORITHMS=( gs rs ts ga hc sa bayes )

# Window sizes to run (space-separated subset of: 100 200 500)
WINDOWS=( 100 200 500 )

# Number of trials per job
N_TRIALS=20

# Base training epochs (used when epochs are not the thing being optimised)
BASE_EPOCHS=30

# Random seed
SEED=42

# Path to the SLURM launcher script
LAUNCHER=/home/hpc/users/ml_models/rebeka.maneva/nlp/code/run_optimization.sh

# Dry run: set to 1 to only print commands, 0 to actually submit
DRY_RUN="${DRY_RUN:-0}"

# =============================================================================
# SUBMISSION LOOP
# =============================================================================

JOB_COUNT=0

for MODEL in "${MODELS[@]}"; do
    for WINDOW in "${WINDOWS[@]}"; do
        for OPT_TYPE in "${OPT_TYPES[@]}"; do
            for ALGORITHM in "${ALGORITHMS[@]}"; do

                JOB_NAME="opt_${MODEL}_${WINDOW}bp_${OPT_TYPE}_${ALGORITHM}"

                CMD=(
                    sbatch
                    --job-name="${JOB_NAME}"
                    --export="ALL,MODEL=${MODEL},OPT_TYPE=${OPT_TYPE},ALGORITHM=${ALGORITHM},N_TRIALS=${N_TRIALS},BASE_EPOCHS=${BASE_EPOCHS},SEED=${SEED},WINDOW=${WINDOW}"
                    "${LAUNCHER}"
                )

                if [ "${DRY_RUN}" = "1" ]; then
                    echo "[DRY RUN] ${CMD[*]}"
                else
                    "${CMD[@]}"
                    echo "Submitted: ${JOB_NAME}"
                fi

                JOB_COUNT=$(( JOB_COUNT + 1 ))

            done
        done
    done
done

echo ""
echo "Total jobs: ${JOB_COUNT}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "(dry run — nothing submitted)"
fi
