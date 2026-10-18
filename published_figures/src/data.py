"""Readers for the results directory (results/revision_2026_10/); nothing outside it is read."""

import sqlite3
from pathlib import Path

import pandas as pd

from . import config as C


def load_results(results: Path) -> pd.DataFrame:
    """Test AUROC of every variant x dataset x classification head of the benchmark (53 variants, 25 datasets)."""
    with sqlite3.connect(results / C.RESULTS_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM classificationreport", conn)
    df = df[df["dataset"].isin(C.DATASETS) & df["embedder"].isin(C.MODELS)]

    if df.duplicated(["dataset", "embedder", "model"]).any():
        raise ValueError(f"Duplicate result rows:\n{df[df.duplicated(['dataset', 'embedder', 'model'])]}")
    full = pd.MultiIndex.from_product([list(C.MODELS), C.DATASETS, list(C.HEADS)])
    missing = full.difference(pd.MultiIndex.from_frame(df[["embedder", "dataset", "model"]]))
    if len(missing):
        raise ValueError(f"Missing results (embedder, dataset, head): {list(missing)}")
    return df


def best_head(df: pd.DataFrame) -> pd.DataFrame:
    """Best classification head (highest test AUROC) per embedder and dataset."""
    return df.groupby(["embedder", "dataset"])["test_metric"].max().reset_index()


def load_posthoc(results: Path) -> pd.DataFrame:
    """Per-fold AUROC of CheMeleon and ECFP count, every head, on the 11 external datasets (scaffold 5-fold CV)."""
    df = pd.read_csv(results / C.POSTHOC_CSV)
    df = df[df["embedder"].isin(C.POSTHOC_MODELS)]
    counts = df.groupby(["embedder", "model"]).agg(n=("auroc", "size"), datasets=("dataset", "nunique"))
    if df.duplicated(["dataset", "embedder", "model", "fold"]).any() or counts["n"].nunique() != 1 \
            or len(counts) != len(C.POSTHOC_MODELS) * len(C.HEADS) or set(df["dataset"]) != set(C.POSTHOC_DATASETS):
        raise ValueError(f"Incomplete posthoc results:\n{counts}")
    return df


def load_runtime(results: Path, keep_all: bool = False) -> pd.DataFrame:
    """Embedding times on MUV; the file is append-only, so the last row of a re-measured embedder wins."""
    rt = pd.read_csv(results / C.RUNTIME_CSV)
    return rt if keep_all else rt.drop_duplicates(["dataset", "embedder", "device", "n_samples"], keep="last")


def load_benchmark_datasets(results: Path) -> pd.DataFrame:
    """Label counts per benchmark dataset and task (after removing the illegal SMILES)."""
    return pd.read_csv(results / C.BENCHMARK_DATASETS_CSV)


def load_external_labels(results: Path, dataset: str) -> pd.Series:
    return pd.read_csv(results / C.EXTERNAL_DATASETS_DIR / f"{dataset}.csv")["label"]
