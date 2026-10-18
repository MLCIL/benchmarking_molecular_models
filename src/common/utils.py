import logging as log
import os

from typing import Optional
from tqdm.auto import trange


TORCH_GPU_ENV = "TORCH_GPU"
EMBED_CPU_NJOBS_ENV = "EMBED_CPU_NJOBS"
DEFAULT_EMBED_CPU_FRACTION = 0.75


def batch(data, n=128):
    data_len = len(data)
    for ndx in trange(0, data_len, n):
        batch = data[ndx:min(ndx + n, data_len)]
        if type(batch) is not list:
            batch = batch.tolist()
        # log.debug(f"Type: {type(batch)} Batch: {batch}")
        yield batch


def get_embed_cpu_njobs(default_fraction: float = DEFAULT_EMBED_CPU_FRACTION) -> int:
    """
    Resolve CPU worker count for embedding (multiprocessing / joblib / Mordred).

    Priority:
      1. ``EMBED_CPU_NJOBS`` env var:
         - integer (e.g. ``128``) → absolute worker count
         - float in ``(0, 1]`` (e.g. ``0.75``) → fraction of ``os.cpu_count()``
      2. ``default_fraction`` of all CPUs (75% by default)

    Always returns an int in ``[1, cpu_count]``.
    """
    total = os.cpu_count() or 1
    raw = os.environ.get(EMBED_CPU_NJOBS_ENV)

    if raw is None or not raw.strip():
        n_jobs = int(total * default_fraction)
    else:
        value = raw.strip()
        try:
            # Integer literals are absolute counts; floats are fractions when in (0, 1].
            if _is_int_literal(value):
                n_jobs = int(value)
            else:
                fraction = float(value)
                if not (0.0 < fraction <= 1.0):
                    raise ValueError(
                        f"{EMBED_CPU_NJOBS_ENV}={value!r}: float must be in (0, 1]"
                    )
                n_jobs = int(total * fraction)
        except ValueError as e:
            raise ValueError(
                f"Invalid {EMBED_CPU_NJOBS_ENV}={value!r}; "
                f"expected an integer (e.g. 128) or a fraction in (0, 1] (e.g. 0.75)"
            ) from e

    n_jobs = max(1, min(n_jobs, total))
    log.info(
        f"Using {n_jobs}/{total} CPU workers for embedding"
        + (f" ({EMBED_CPU_NJOBS_ENV}={raw!r})" if raw is not None else
           f" (default {default_fraction:.0%} of CPUs)")
    )
    return n_jobs


def _is_int_literal(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


try:
    import torch
    import subprocess
    import pandas as pd
    import sys
    from io import StringIO
    
    def cuda_available() -> bool:
        return torch.cuda.is_available()
    
    def get_least_utilized_gpu() -> int:
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode(sys.stdout.encoding)
        gpu_df = pd.read_csv(StringIO(gpu_stats),
                            names=['memory.used', 'memory.free'],
                            skiprows=1)
        log.debug('GPU usage:\n{}'.format(gpu_df))
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].astype(float).idxmax()
        log.info('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        return idx

    def _normalize_torch_gpu_env(value: str) -> str:
        """Normalize TORCH_GPU values like 'cpu', 'cuda', 'cuda:1', or bare '1'."""
        value = value.strip()
        if not value:
            raise ValueError(f"{TORCH_GPU_ENV} is set but empty")
        lower = value.lower()
        if lower == "cpu":
            return "cpu"
        if lower.startswith("cuda"):
            return lower
        if value.isdigit():
            return f"cuda:{value}"
        raise ValueError(
            f"Invalid {TORCH_GPU_ENV}={value!r}; expected 'cpu', 'cuda', 'cuda:N', or a GPU index"
        )
    
    def get_device(device: Optional[str] = None, optimize_gpu_distribution: bool = True) -> torch.device:
        """
        Resolve the torch device.

        Priority:
          1. Explicit ``device`` argument
          2. ``TORCH_GPU`` env var (e.g. ``TORCH_GPU=cuda:1`` or ``TORCH_GPU=cpu``)
          3. Auto-select least-utilized GPU when CUDA is available, else CPU
        """
        env_raw = os.environ.get(TORCH_GPU_ENV)

        if device is None and env_raw is not None:
            device = _normalize_torch_gpu_env(env_raw)
            if device == "cpu":
                log.info(f"{TORCH_GPU_ENV}={env_raw!r}: forcing CPU (auto GPU selection disabled)")
            elif not torch.cuda.is_available():
                log.warning(
                    f"{TORCH_GPU_ENV}={env_raw!r} requested GPU but CUDA is unavailable; falling back to CPU"
                )
                device = "cpu"
            else:
                log.info(
                    f"{TORCH_GPU_ENV}={env_raw!r}: using device {device} "
                    f"(auto GPU selection disabled)"
                )
            return torch.device(device)

        if device is None:
            if torch.cuda.is_available():
                if optimize_gpu_distribution:
                    device = f"cuda:{get_least_utilized_gpu()}"
                else:
                    device = "cuda"
            else:
                device = "cpu"
            log.warning(f"Using device: {device}")

        return torch.device(device)
except Exception as e:
    log.info(e)
