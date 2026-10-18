#!/bin/bash

# embed_wrapper.sh cds into this directory before sourcing init.sh
SCRIPT_DIR="$(pwd)"
CKPT_DIR="$SCRIPT_DIR/ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints"
MODEL_PTH="$CKPT_DIR/model.pth"
CONFIG_YAML="$CKPT_DIR/config.yaml"
ZENODO_MODEL="https://zenodo.org/records/10803099/files/model.pth?download=1"
ZENODO_CONFIG="https://zenodo.org/records/10803099/files/config.yaml?download=1"
INSTALL_DEP=${INSTALL_DEP:-false}

download_weights() {
  mkdir -p "$CKPT_DIR"

  if [ ! -f "$CONFIG_YAML" ]; then
    echo "Downloading MolE config.yaml from Zenodo..."
    if ! curl -fL --retry 3 --retry-delay 2 -o "$CONFIG_YAML" "$ZENODO_CONFIG"; then
      cat <<'EOF'

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ERROR: Failed to download MolE config.yaml from Zenodo.
Please download manually and place it at:
  model_wrappers/mole/ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints/config.yaml

  curl -o model_wrappers/mole/ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints/config.yaml \
    "https://zenodo.org/records/10803099/files/config.yaml?download=1"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

EOF
      return 1
    fi
  fi

  if [ ! -f "$MODEL_PTH" ]; then
    echo "Downloading MolE model.pth from Zenodo (~804 MB)..."
    if ! curl -fL --retry 3 --retry-delay 2 -o "$MODEL_PTH" "$ZENODO_MODEL"; then
      rm -f "$MODEL_PTH"
      cat <<'EOF'

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ERROR: Failed to download MolE model.pth from Zenodo (~804 MB).
Please download manually and place it at:
  model_wrappers/mole/ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints/model.pth

  curl -o model_wrappers/mole/ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints/model.pth \
    "https://zenodo.org/records/10803099/files/model.pth?download=1"
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

EOF
      return 1
    fi
  else
    echo "MolE weights already present at $MODEL_PTH"
  fi
}

if $INSTALL_DEP; then
  # Clone upstream MolE sources (models + graph featurization)
  if [ ! -d "$SCRIPT_DIR/mole_repo/.git" ]; then
    rm -rf "$SCRIPT_DIR/mole_repo"
    git clone --depth 1 https://github.com/rolayoalarcon/MolE.git "$SCRIPT_DIR/mole_repo"
  fi

  # Match upstream MolE stack (CPU-friendly; CUDA wheels if available)
  pip install \
    'torch>=2.1,<2.5' \
    'torch_geometric>=2.5.0' \
    'rdkit>=2022.3.3' \
    'PyYAML>=6.0.1' \
    'pandas>=2.0' \
    'numpy>=1.24,<2.0'

  if ! download_weights; then
    exit 1
  fi
else
  # Still ensure weights exist on subsequent runs
  if [ ! -f "$MODEL_PTH" ] || [ ! -f "$CONFIG_YAML" ]; then
    if ! download_weights; then
      exit 1
    fi
  fi
fi

export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/mole_repo"
export HYDRA_EXPERIMENT=mole
