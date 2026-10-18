import logging as log
import numpy as np
import os
import joblib
import sys

from hydra.utils import get_original_cwd
from os.path import join
from typing import Dict, List, Optional
from ..common.types import EmbeddingConfig, Dataset, EmbeddedDataset, Embedder
from time import time


RUNTIME_CSV_FIELDS = [
    "dataset",
    "embedder",
    "device",
    "mean_runtime",
    "std_runtime",
    "total_runtime",
    "n_samples",
    "n_full_batches",
]


def is_already_embedded(config: EmbeddingConfig, dataset_name: str, model: Embedder) -> bool:
    model_name = model.name
    output_dir = join(get_original_cwd(), config.embedded_directory, dataset_name)
    embedding_file = join(output_dir, f"{model_name}.joblib")
    if sys.version_info < (3, 9):
        return os.path.exists(embedding_file.replace(".joblib", ".json"))
    return os.path.exists(embedding_file)


def normalize_device(device: str) -> str:
    device = str(device).lower()
    if "cuda" in device or device.startswith("gpu"):
        return "cuda"
    return "cpu"


def _runtime_csv_path(config: EmbeddingConfig) -> str:
    return join(get_original_cwd(), config.runtime_csv)


def _clock_embedding_path(
    config: EmbeddingConfig, dataset_name: str, model_name: str, device: str
) -> str:
    output_dir = join(get_original_cwd(), config.clock_directory, dataset_name)
    suffix = ".joblib" if sys.version_info >= (3, 9) else ".json"
    return join(output_dir, f"{model_name}__{device}{suffix}")


def _read_runtime_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r") as f:
        header_line = f.readline()
        if not header_line:
            return []
        fields = header_line.rstrip("\n").split(",")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split(",")
            rows.append({fields[i]: values[i] if i < len(values) else "" for i in range(len(fields))})
    return rows


def _find_runtime_row(
    config: EmbeddingConfig,
    dataset_name: str,
    embedder: str,
    device: str,
    batch_size: int,
) -> Optional[Dict[str, str]]:
    device = normalize_device(device)
    for row in _read_runtime_csv(_runtime_csv_path(config)):
        if (
            row.get("dataset") == dataset_name
            and row.get("embedder") == embedder
            and normalize_device(row.get("device", "")) == device
            and str(row.get("n_samples", "")) == str(batch_size)
        ):
            return row
    return None


def append_runtime_csv(config: EmbeddingConfig, row: Dict) -> str:
    """Append one measurement row to the shared runtime CSV (create with header if needed)."""
    path = _runtime_csv_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(RUNTIME_CSV_FIELDS) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in RUNTIME_CSV_FIELDS) + "\n")
    return path


def is_already_measured(
    config: EmbeddingConfig,
    dataset_name: str,
    model: Embedder,
    device: str,
    batch_size: int,
) -> bool:
    device = normalize_device(device)
    if _find_runtime_row(config, dataset_name, model.name, device, batch_size) is None:
        return False
    return os.path.exists(_clock_embedding_path(config, dataset_name, model.name, device))


def measure_embedding(
    config: EmbeddingConfig,
    dataset: Dataset,
    model: Embedder,
    batch_size: int = 10_000,
    cache: bool = True,
) -> Dict[str, float]:
    """
    Embed ``dataset`` in chunks of ``batch_size``, recording per-batch and total runtime.

    Mean/std are computed only over *full* batches of ``batch_size`` (the trailing
    partial batch is excluded from those stats but included in ``total_runtime``).
    Embeddings are written under ``config.clock_directory``; timings append to
    ``config.runtime_csv`` (stdlib CSV — no peewee in model venvs).
    """
    device = normalize_device(model.device_used)
    model_name = model.name
    log_header = f"Model : {model_name: <20}, dataset: {dataset.name: <20}, device: {device: <5} |"

    if cache and is_already_measured(config, dataset.name, model, device, batch_size):
        log.info(f"{log_header} measurement already exists, skipping")
        row = _find_runtime_row(config, dataset.name, model_name, device, batch_size)
        return {
            "mean_runtime": float(row["mean_runtime"]),
            "std_runtime": float(row["std_runtime"]),
            "total_runtime": float(row["total_runtime"]),
            "device": device,
            "n_full_batches": int(float(row.get("n_full_batches") or 0)),
        }

    n = len(dataset.data)
    batch_times = []
    embeddings = []

    total_start = time()
    for start_idx in range(0, n, batch_size):
        # Copy so SmilesEmbedder wildcard fixes do not hit a SettingWithCopy view.
        batch_df = dataset.data.iloc[start_idx : start_idx + batch_size].copy()
        is_full = len(batch_df) == batch_size

        t0 = time()
        batch_emb = model.embed(batch_df)
        elapsed = time() - t0

        embeddings.append(batch_emb)
        if is_full:
            batch_times.append(elapsed)
            log.info(
                f"{log_header} full batch {start_idx // batch_size + 1} "
                f"({batch_size} samples) took {elapsed:.2f}s"
            )
        else:
            log.info(
                f"{log_header} trailing batch ({len(batch_df)} samples) took {elapsed:.2f}s "
                f"(excluded from mean/std)"
            )

    total_runtime = time() - total_start

    if batch_times:
        mean_runtime = float(np.mean(batch_times))
        std_runtime = float(np.std(batch_times))
    else:
        mean_runtime = float("nan")
        std_runtime = float("nan")
        log.warning(
            f"{log_header} no full batches of size {batch_size}; mean/std are NaN"
        )

    log.info(
        f"{log_header} mean={mean_runtime:.2f}s ± {std_runtime:.2f}s per {batch_size}, "
        f"total={total_runtime:.2f}s ({n} samples, {len(batch_times)} full batches)"
    )

    X = np.concatenate(embeddings, axis=0)
    embedded_dataset = EmbeddedDataset(
        name=dataset.name,
        task=dataset.task,
        embedder=model_name,
        splits=dataset.splits,
        X=X,
        y=dataset.labels,
    )
    invalid_embeds = embedded_dataset.remove_failed_embeddings()
    if invalid_embeds > 0:
        log.warning(
            f"{log_header} removed {invalid_embeds} invalid embeddings from '{dataset.name}'"
        )
        if invalid_embeds > config.max_invalid_embeddings:
            raise ValueError(
                f"Too many invalid embeddings ({invalid_embeds}) for dataset "
                f"'{dataset.name}' with model '{model_name}'"
            )

    embedding_file = _clock_embedding_path(config, dataset.name, model_name, device)
    os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
    if sys.version_info < (3, 9):
        embedded_dataset.serialize_legacy(embedding_file)
    else:
        joblib.dump(embedded_dataset, embedding_file)
    log.info(f"{log_header} saved embeddings to {embedding_file}")

    csv_path = append_runtime_csv(
        config,
        {
            "dataset": dataset.name,
            "embedder": model_name,
            "device": device,
            "mean_runtime": mean_runtime,
            "std_runtime": std_runtime,
            "total_runtime": total_runtime,
            "n_samples": batch_size,
            "n_full_batches": len(batch_times),
        },
    )
    log.info(f"{log_header} appended timings to {csv_path}")

    del embeddings, embedded_dataset, X
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    return {
        "mean_runtime": mean_runtime,
        "std_runtime": std_runtime,
        "total_runtime": total_runtime,
        "device": device,
        "n_full_batches": len(batch_times),
    }


def clock(
    config: EmbeddingConfig,
    dataset: Dataset,
    model: Embedder,
    batch_size: int = 10_000,
    cache: bool = True,
) -> Dict[str, float]:
    """Alias for :func:`measure_embedding` (legacy name used by ``--clock``)."""
    return measure_embedding(config, dataset, model, batch_size=batch_size, cache=cache)


def embed(config: EmbeddingConfig, dataset: Dataset, model: Embedder, cache: bool = True) -> None:
    model_name = model.name
    output_dir = join(get_original_cwd(), config.embedded_directory, dataset.name)
    embedding_file = join(output_dir, f"{model_name}.joblib" if sys.version_info >= (3, 9) else f"{model_name}.json")
    log_header = f"Model : {model_name: <20}, dataset: {dataset.name: <20} |"
    
    if cache and os.path.exists(embedding_file):
        log.info(f"Loading cached embeddings {embedding_file}")
        return joblib.load(embedding_file)
    elif not cache:
        log.info("Cache disabled, recomputing embeddings")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start = time()
    embeddings = model.embed(dataset.data)
    runtime = time() - start
    
    log.info(f"{log_header} took {runtime:.2f} seconds")

    embedded_dataset = EmbeddedDataset(
        name=dataset.name,
        task=dataset.task,
        embedder=model_name,
        splits=dataset.splits,
        X=embeddings,
        y=dataset.labels
    )
    
    invalid_embeds = embedded_dataset.remove_failed_embeddings()
    if invalid_embeds > 0:
        log.warning(f"{log_header} removed {invalid_embeds} invalid embeddings from the dataset '{dataset.name}'")
        if invalid_embeds > config.max_invalid_embeddings:
            log.error(f"{log_header} too many invalid embeddings ({invalid_embeds}), stopping embedding process")
            raise ValueError(f"Too many invalid embeddings ({invalid_embeds}) for dataset '{dataset.name}' with model '{model_name}'")
    

    if sys.version_info < (3, 9):
        embedded_dataset.serialize_legacy(embedding_file)
    else:
        joblib.dump(embedded_dataset, embedding_file)

    # EmbeddingMeta.create(
    #     dataset=dataset.name,
    #     embedder=model_name,
    #     embedding_loc=embedding_file,
    #     embedding_time=runtime,    
    # )
    
    log.info(f"{log_header} saved results")
    
    del embeddings, embedded_dataset, model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def load_embedding(config: EmbeddingConfig, dataset_name: str, model_name: str) -> EmbeddedDataset:
    output_dir = join(get_original_cwd(), config.embedded_directory, dataset_name)
    embedding_file = join(output_dir, f"{model_name}.joblib")
    return joblib.load(embedding_file)
