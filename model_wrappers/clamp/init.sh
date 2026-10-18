#!/bin/bash

if $INSTALL_DEP; then
    # clamp_repo's setup.py needs pkg_resources, which setuptools>=81 no
    # longer bundles. Pinning setuptools in the venv isn't enough on its own:
    # pip's default build isolation fetches its own (latest) setuptools for
    # the build step, so also pass --no-build-isolation to make it use ours.
    pip install "setuptools<81"
    pip install --no-build-isolation -e ./clamp_repo
    # CLAMP_refitted weights come from the Hugging Face Hub
    pip install huggingface_hub
fi

export HYDRA_EXPERIMENT=clamp
