#!/bin/bash

if $INSTALL_DEP; then
  # Community-maintained Mordred (original package is unmaintained / broken on modern NumPy)
  pip install 'mordredcommunity[full]'
fi

export HYDRA_EXPERIMENT=mordred
