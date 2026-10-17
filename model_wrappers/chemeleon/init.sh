#!/bin/bash

if $INSTALL_DEP; then
    pip install 'chemprop>=2.2.0'

    # Pre-download CheMeleon message-passing weights (also fetched lazily by CheMeleonFingerprint)
    CKPT_DIR="${HOME}/.chemprop"
    mkdir -p "$CKPT_DIR"
    if [ ! -f "$CKPT_DIR/chemeleon_mp.pt" ]; then
        echo "Downloading CheMeleon weights to $CKPT_DIR/chemeleon_mp.pt"
        curl -fsSL -o "$CKPT_DIR/chemeleon_mp.pt" \
            "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    fi
fi

export HYDRA_EXPERIMENT=chemeleon
