#!/bin/bash

# Non-interactive shells (nohup, cron, this-tool's Bash) don't source ~/.bashrc,
# so `uv` (installed under ~/.local/bin) would otherwise be missing from PATH.
export PATH="$HOME/.local/bin:$HOME/.pyenv/bin:$PATH"

# Pin the GPU by masking rather than by asking nicely. Several wrappers
# (chemformer, clamp) resolve the right device and then hand their own
# trainer a bare "cuda"/n_gpus=1, which always lands on GPU 0 and ignores
# TORCH_GPU. Hiding the other GPUs makes the pin hold regardless, and the
# requested GPU becomes cuda:0 inside the masked process.
if [[ "${TORCH_GPU:-}" =~ ^cuda:([0-9]+)$ ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
    export TORCH_GPU="cuda:0"
fi

# parse
DELETE=false
CACHE=true
CMD=embed.py
REINSTALL=""
WRAPPER_PATH=""
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--delete)
            DELETE=true
            shift
            ;;
        -r|--reinstall)
            REINSTALL=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --clock|--measure)
            CMD=measure_embedding.py
            shift
            ;;
        --embed)
            CMD=embed.py
            shift
            ;;
        --)
            shift
            HYDRA_ARGS+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Hydra overrides go after the wrapper path, e.g.:"
            echo "  ./embed_wrapper.sh model_wrappers/foo 'hydra.sweeper.params.+dataset=glob(clf_ogbg-molbbbp)'"
            exit 1
            ;;
        *)
            if [ -z "$WRAPPER_PATH" ]; then
                WRAPPER_PATH=$1
            else
                HYDRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

run() {
    WRAPPER_ABS=$(cd "$WRAPPER_PATH" && pwd)
    VENV_PY="$WRAPPER_ABS/venv/bin/python"
    cd "$WRAPPER_ABS"

    # install environment
    if [ -d "venv" ] && [ -z "$REINSTALL" ]; then
        export INSTALL_DEP=false
    else
        export INSTALL_DEP=true
        python_version=$(cat .python-version 2>/dev/null)
        python_version=${python_version:-3.11}
        PY_BIN="python$python_version"
        # uv only finds bare "pythonX.Y" on PATH; EOL versions (e.g. 3.7) aren't
        # in uv's managed registry but may exist as a pyenv install, so fall
        # back to that install's absolute interpreter path.
        if ! command -v "$PY_BIN" >/dev/null 2>&1 && command -v pyenv >/dev/null 2>&1; then
            PYENV_CANDIDATE=$(pyenv versions --bare 2>/dev/null | grep -E "^${python_version}(\.[0-9]+)?$" | sort -V | tail -1)
            if [ -n "$PYENV_CANDIDATE" ]; then
                PY_BIN="$(pyenv root)/versions/$PYENV_CANDIDATE/bin/python$python_version"
            fi
        fi
        if uv venv venv --seed --python="$PY_BIN"; then
            uv pip install --python "$VENV_PY" -r ../../base_requirements.txt || return 1
        else
            # uv supports only Python >=3.7. Older wrappers (cddd pins 3.6) build
            # through the interpreter's own venv module and its bundled pip.
            "$PY_BIN" -m venv venv || return 1
            "$VENV_PY" -m pip install -r ../../base_requirements.txt || return 1
        fi
    fi

    # init.sh may call pip; prefer venv binaries without requiring an activated shell
    export PATH="$WRAPPER_ABS/venv/bin:$PATH"
    source ./init.sh || return $?

    cd ../../

    export PYTHONPATH=$PYTHONPATH:.:$WRAPPER_PATH

    # Fail loudly rather than silently reporting "Done" if venv setup didn't
    # actually produce a usable interpreter (e.g. uv missing/failed above).
    if [ ! -x "$VENV_PY" ]; then
        echo "ERROR: venv python not found at $VENV_PY" >&2
        return 1
    fi

    # Invoke the model venv python directly. `uv run --python <venv>` still
    # prefers the project .venv when one exists, so chemprop/etc. would be missing.
    "$VENV_PY" -u $CMD --multirun \
        +experiment=$HYDRA_EXPERIMENT ++cache=$CACHE \
        "${HYDRA_ARGS[@]}"
    status=$?

    echo "Done"
    return $status
}

delete() {
    rm -rf $WRAPPER_PATH/venv
}

if [ -z "$WRAPPER_PATH" ]; then
    echo "Please provide the path to the wrapper"
    exit 1
fi

if $DELETE; then
    delete
else
    run
fi
