#!/bin/bash

if $INSTALL_DEP; then
    pip install unimol_tools --upgrade
    pip install huggingface_hub
    # No pandas/numpy pin here on purpose. This used to force "pandas<2"
    # (plus "numpy<2" for its ABI), but unimol_tools itself requires
    # pandas>=2.2.2 -- the pin contradicted the package's own metadata and
    # made this venv the only one writing pandas 1.x pickles, which the
    # pandas 2.x scoring venv then could not unpickle at all.
    # See src/common/pandas_compat.py for reading artifacts already written
    # under the old pin.
fi

export HYDRA_EXPERIMENT=unimol
