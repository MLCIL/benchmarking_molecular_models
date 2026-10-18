# =============================================================================
# WARNING: Do not call directly, only use via bash scripts
#   ./embed_wrapper.sh --clock model_wrappers/<model> ...
#   ./run_measure_muv.sh
# =============================================================================

import joblib
import os
import sys
import hydra
import logging as log

from time import sleep
from os.path import join
from hydra.utils import get_original_cwd
from src.common.types import EmbeddingConfig, Dataset
from src.common.utils import TORCH_GPU_ENV
from src.embedding.embedding import (
    measure_embedding,
    is_already_measured,
    normalize_device,
)
from wrapper import get_embedder


def _load_dataset(cfg, embed_config: EmbeddingConfig) -> Dataset:
    dataset_name = cfg.dataset.name
    dataset_path = join(
        get_original_cwd(), embed_config.prepared_directory, f"{dataset_name}.joblib"
    )
    if sys.version_info < (3, 9):
        log.info("Using legacy loader")
        return Dataset.deserialize_legacy(dataset_path.replace(".joblib", ".json"))
    log.info("Using joblib loader")
    return joblib.load(dataset_path)


def _filter_illegal(cfg, dataset: Dataset) -> None:
    illegal_smiles_path = cfg.illegal_smiles if "illegal_smiles" in cfg else None
    if illegal_smiles_path is None:
        return
    with open(join(get_original_cwd(), illegal_smiles_path), "r") as f:
        illegal_smiles = [line.strip() for line in f.readlines()]
    log.info(f"Filtering out illegal SMILES: {illegal_smiles}")
    dataset.filter_out_problematic_molecules(illegal_smiles)


def _build_embedder(cfg):
    if "model_name" in cfg:
        model_name = cfg.model_name
    else:
        model_name = cfg.model.model_name
    kwargs = cfg.model.kwargs if "model" in cfg and "kwargs" in cfg.model else {}
    return get_embedder(model_name, task=cfg.dataset.task, **kwargs)


def _free_model(model) -> None:
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _should_also_measure_cpu(device: str, cfg) -> bool:
    """GPU runs also get a CPU pass unless the caller already forced CPU."""
    if normalize_device(device) != "cuda":
        return False
    if not cfg.get("measure_cpu_when_gpu", True):
        return False
    # Caller already requested CPU-only via env — do not schedule a second CPU run.
    env = os.environ.get(TORCH_GPU_ENV, "").strip().lower()
    return env != "cpu"


@hydra.main(config_path="./config", config_name="clock")
def main(cfg):
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    embed_config = EmbeddingConfig(**cfg.embedding)
    batch_size = int(cfg.get("batch_size", 10_000))
    cache = bool(cfg.cache) if "cache" in cfg else True

    dataset = _load_dataset(cfg, embed_config)
    _filter_illegal(cfg, dataset)

    model = _build_embedder(cfg)
    device = normalize_device(model.device_used)
    log.info(f"Measuring model: {model.name} on device={device}")

    if cache and is_already_measured(
        embed_config, dataset.name, model, device, batch_size
    ):
        log.info(
            f"Measurement already exists for {model.name} on {dataset.name} ({device})"
        )
    else:
        measure_embedding(
            embed_config, dataset, model, batch_size=batch_size, cache=cache
        )

    also_cpu = _should_also_measure_cpu(device, cfg)
    _free_model(model)

    if also_cpu:
        log.info(f"Re-running {cfg.dataset.name} measurement on CPU")
        os.environ[TORCH_GPU_ENV] = "cpu"
        model_cpu = _build_embedder(cfg)
        assert normalize_device(model_cpu.device_used) == "cpu", (
            f"Expected CPU embedder after {TORCH_GPU_ENV}=cpu, "
            f"got {model_cpu.device_used}"
        )
        measure_embedding(
            embed_config, dataset, model_cpu, batch_size=batch_size, cache=cache
        )
        _free_model(model_cpu)

    sleep(5)


if __name__ == "__main__":
    main()
