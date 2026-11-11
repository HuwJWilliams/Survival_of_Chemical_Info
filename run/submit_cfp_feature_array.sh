#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --account=palmer-addnm
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=cfp_feature
#SBATCH --output=/users/yhb18174/TL_project/run/slurm_files/cfp_feature-%A_%a.out

set -euo pipefail

PROJECT_ROOT="/users/yhb18174/TL_project"

module purge
module load miniforge/python-3.12.10/25.3.0
/opt/software/scripts/job_prologue.sh
trap '[[ -f /opt/software/scripts/job_epilogue.sh ]] && /opt/software/scripts/job_epilogue.sh' EXIT

mamba activate tlp_py312
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Keep this list in sync with the --array range supplied to sbatch.
# Each name must be supported by config.py and mapped in paths.json for "all".
TRAIN_FEATURES=(
    molformer-dc
    ft-random-molformer-dc
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Submit this script with sbatch --array=0-$(( ${#TRAIN_FEATURES[@]} - 1 ))" >&2
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TRAIN_FEATURES[@]} )); then
    echo "Array index $SLURM_ARRAY_TASK_ID is outside the feature list" >&2
    exit 2
fi

TRAIN_NAME="${TRAIN_FEATURES[$SLURM_ARRAY_TASK_ID]}"
SAVE_DIR="${SAVE_DIR:-cross_feature_predictions}"
N_RESAMPLES="${N_RESAMPLES:-10}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.3}"
MINIMUM_TARGS="${MINIMUM_TARGS:-2500}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

ARGS=(
    "$PROJECT_ROOT/run/run_cfp.py"
    --property all
    --train "$TRAIN_NAME"
    --target mordred
    --model-name random_forest
    --save-dir "$SAVE_DIR"
    --n-resamples "$N_RESAMPLES"
    --random-seed "$RANDOM_SEED"
    --test-size "$TEST_SIZE"
    --minimum-targs "$MINIMUM_TARGS"
    --save-models
    --save-feat-imp
)

if [[ "$SKIP_EXISTING" == "true" ]]; then
    ARGS+=(--skip-existing)
fi

echo "Array task $SLURM_ARRAY_TASK_ID: pred_mordred_tr_${TRAIN_NAME}"
echo "Resamples: $N_RESAMPLES | Seed: $RANDOM_SEED | Save dir: $SAVE_DIR"
python -u "${ARGS[@]}"
