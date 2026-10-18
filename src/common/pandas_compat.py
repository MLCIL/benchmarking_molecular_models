"""Read embedding artifacts pickled by pandas 1.x while running pandas 2.x.

Model wrappers run in their own venvs, and some are pinned to ``pandas<2``
(unimol, for one, via its ``init.sh``). A DataFrame pickled there stores a
reference to ``pandas.core.indexes.numeric`` -- a module pandas 2.0 deleted
after folding Int64Index/Float64Index into the generic ``Index``.

Scoring runs in the project venv on pandas 2.x, so loading those files dies
with ``ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'``
inside pickle, before any of our code gets a chance to intervene. The only
fix that does not require re-embedding is to make the module importable
again, which is what this shim does.
"""

import logging as log
import sys
import types


_SHIM_MODULE = "pandas.core.indexes.numeric"


def install_pandas1_unpickle_shim() -> bool:
    """Register a stand-in for the index module pandas 2 removed.

    Safe to call repeatedly, and a no-op on pandas 1.x where the real module
    still exists. Returns True when a shim is in place.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        return False

    try:
        __import__(_SHIM_MODULE)
        return False  # real module present (pandas 1.x): nothing to do
    except ImportError:
        pass

    if _SHIM_MODULE in sys.modules:
        return True

    module = types.ModuleType(_SHIM_MODULE)
    # Every one of these became a plain Index in pandas 2; unpickling routes
    # through Index._new_Index, so the generic class reconstructs them fine.
    for attr in ("Int64Index", "UInt64Index", "Float64Index", "NumericIndex"):
        setattr(module, attr, pd.Index)
    sys.modules[_SHIM_MODULE] = module
    log.debug(
        "Installed %s shim so pandas 1.x embedding pickles stay readable",
        _SHIM_MODULE,
    )
    return True
