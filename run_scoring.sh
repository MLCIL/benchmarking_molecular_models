#!/bin/bash

# Non-interactive shells (nohup, cron) don't source ~/.bashrc, so `uv`
# (installed under ~/.local/bin) would otherwise be missing from PATH.
export PATH="$HOME/.local/bin:$HOME/.pyenv/bin:$PATH"

DATETIME=$(date '+%Y%m%d_%H%M')

# Experiments come first; anything after `--` is passed through to hydra, e.g.
#   ./run_scoring.sh clamp -- '+model=clamp_refitted'
EXPERIMENTS=()
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ $1 == "--" ]]; then
        shift
        HYDRA_ARGS+=("$@")
        break
    fi
    EXPERIMENTS+=("$1")
    shift
done

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "Usage: $0 <experiment> [<experiment>...] [-- <hydra overrides>]"
    exit 1
fi

mkdir -p logs_scoring

nohup bash -c '
  DATETIME="$1"; shift
  N_HYDRA="$1"; shift
  HYDRA_ARGS=("${@:1:$N_HYDRA}")
  shift "$N_HYDRA"
  for EXPERIMENT in "$@"; do
    echo "Starting experiment: ${EXPERIMENT}"
    nice -n 10 uv run python score.py --multirun +experiment=${EXPERIMENT} \
      "${HYDRA_ARGS[@]}" \
      > logs_scoring/${EXPERIMENT}_${DATETIME}.log 2>&1
    echo "Finished experiment: ${EXPERIMENT} (exit $?)"
  done
' _ "$DATETIME" "${#HYDRA_ARGS[@]}" "${HYDRA_ARGS[@]}" "${EXPERIMENTS[@]}" \
  > logs_scoring/batch_${DATETIME}.log 2>&1 &

disown

echo "PID $! started (${#EXPERIMENTS[@]} experiments sequentially, overrides: ${HYDRA_ARGS[*]:-none}, batch log: logs_scoring/batch_${DATETIME}.log)"
