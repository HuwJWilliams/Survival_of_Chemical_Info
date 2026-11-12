#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --account=palmer-addnm
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=cfp_all
#SBATCH --output=/users/yhb18174/TL_project/run/slurm_files/cfp_all-%j.out

PROJECT_ROOT="/users/yhb18174/TL_project"

module purge
module load miniforge/python-3.12.10/25.3.0
/opt/software/scripts/job_prologue.sh
trap '[[ -f /opt/software/scripts/job_epilogue.sh ]] && /opt/software/scripts/job_epilogue.sh' EXIT

mamba activate tlp_py312
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p "$PROJECT_ROOT/run/slurm_files"

SAVE_DIR="${SAVE_DIR:-cross_feature_predictions}"
N_RESAMPLES="${N_RESAMPLES:-10}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TEST_SIZE="${TEST_SIZE:-0.3}"
MINIMUM_TARGS="${MINIMUM_TARGS:-2500}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

# These feature names must be present in both config.py and paths.json.
PROPERTIES=(all)
TRAIN_FEATURES=(molformer-dc)
TARGET_FEATURES=(mordred)
MODELS=(random_forest)

FAILED_RUNS=()

echo "Properties: ${PROPERTIES[*]}"
echo "Train features: ${TRAIN_FEATURES[*]}"
echo "Target features: ${TARGET_FEATURES[*]}"
echo "Models: ${MODELS[*]}"
echo "Save dir: $SAVE_DIR | Resamples: $N_RESAMPLES | Seed: $RANDOM_SEED | Test size: $TEST_SIZE"
echo "Minimum training samples: $MINIMUM_TARGS | Skip existing: $SKIP_EXISTING"

for PROP in "${PROPERTIES[@]}"; do
    for TRAIN_NAME in "${TRAIN_FEATURES[@]}"; do
        for TARGET_NAME in "${TARGET_FEATURES[@]}"; do
            for MODEL_NAME in "${MODELS[@]}"; do
                IDENTIFIER="pred_${TARGET_NAME}_tr_${TRAIN_NAME}"
                ARGS=(
                    "$PROJECT_ROOT/run/run_cfp.py"
                    --property "$PROP"
                    --train "$TRAIN_NAME"
                    --target "$TARGET_NAME"
                    --model-name "$MODEL_NAME"
                    --save-dir "$SAVE_DIR"
                    --n-resamples "$N_RESAMPLES"
                    --random-seed "$RANDOM_SEED"
                    --test-size "$TEST_SIZE"
                    --minimum-targs "$MINIMUM_TARGS"
                )

                if [[ "$MODEL_NAME" == "random_forest" ]]; then
                    ARGS+=(--save-feat-imp)
                fi
                if [[ "$SKIP_EXISTING" == "true" ]]; then
                    ARGS+=(--skip-existing)
                fi

                echo "Running: property=$PROP experiment=$IDENTIFIER model=$MODEL_NAME"
                python -u "${ARGS[@]}"
                RC=$?

                if [[ "$RC" -ne 0 ]]; then
                    echo "FAILED: $PROP / $IDENTIFIER / $MODEL_NAME (exit $RC)"
                    FAILED_RUNS+=("${PROP}:${IDENTIFIER}:${MODEL_NAME}:${RC}")
                else
                    echo "COMPLETED: $PROP / $IDENTIFIER / $MODEL_NAME"
                fi
            done
        done
    done
done

if [[ "${#FAILED_RUNS[@]}" -ne 0 ]]; then
    printf 'Failed runs:\n'
    printf '  %s\n' "${FAILED_RUNS[@]}"
    exit 1
fi

echo "All CFP runs completed."
