#!/bin/bash
# Measure embedding runtime on MUV (ogbg-molmuv) for every model wrapper.
#
# For GPU-capable models, measure_embedding.py records both CUDA and CPU runs
# (via TORCH_GPU) in a single job. Timings append to embedding.runtime_csv
# (data/clock/runtime.csv); embeddings go under embedding.clock_directory.
#
# Usage:
#   ./run_measure_muv.sh                 # sequential, all models
#   ./run_measure_muv.sh --no-cache      # force re-measure (still appends CSV rows)
#   nohup ./run_measure_muv.sh >logs_clock/measure_all_muv.log 2>&1 &

set -u

DATETIME=$(date '+%Y%m%d_%H%M')
# CLI sweep overrides win over hydra.sweeper.params (BasicSweeper).
DATASET_OVERRIDE='+dataset=glob(clf_ogbg-molmuv)'
MASTER_LOG="logs_clock/measure_all_muv_${DATETIME}.log"
EXTRA_ARGS=("$@")
FAILED=()
# Models to leave out of the sweep (e.g. under active debugging).
SKIP=()

mkdir -p logs_clock

echo "Starting all-model MUV runtime measurements at ${DATETIME}" | tee "$MASTER_LOG"

for WRAPPER_DIR in model_wrappers/*/; do
    MODEL=$(basename "$WRAPPER_DIR")
    if [[ ! -f "${WRAPPER_DIR}init.sh" ]]; then
        continue
    fi

    if [[ " ${SKIP[*]} " == *" ${MODEL} "* ]]; then
        echo "=== SKIPPING ${MODEL} ===" | tee -a "$MASTER_LOG"
        continue
    fi

    LOG="logs_clock/measure_${MODEL}_muv_${DATETIME}.log"
    echo "=== Measuring ${MODEL} on MUV ===" | tee -a "$MASTER_LOG"

    if nice -n 10 ./embed_wrapper.sh --clock "${EXTRA_ARGS[@]}" "model_wrappers/${MODEL}" \
        "$DATASET_OVERRIDE" \
        >"$LOG" 2>&1; then
        echo "=== Finished ${MODEL} ===" | tee -a "$MASTER_LOG"
    else
        echo "=== FAILED ${MODEL} (see ${LOG}) ===" | tee -a "$MASTER_LOG"
        FAILED+=("$MODEL")
    fi
done

if ((${#FAILED[@]})); then
    echo "Completed with failures: ${FAILED[*]}" | tee -a "$MASTER_LOG"
    exit 1
fi

echo "All MUV runtime measurements complete." | tee -a "$MASTER_LOG"
echo "Results CSV: data/clock/runtime.csv" | tee -a "$MASTER_LOG"
