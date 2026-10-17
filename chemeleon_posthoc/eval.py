import hydra
import os
import pandas as pd
import numpy as np
import logging as log

from time import time
from filelock import FileLock
from hydra.core.hydra_config import HydraConfig
from tqdm.auto import tqdm
from skfp.fingerprints import ECFPFingerprint, ChemeleonFingerprint
from src.models import get_clf_models, RANDOM_STATE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score


np.random.seed(RANDOM_STATE)


def get_skfp_roc_auc(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # A single target stored as a column looks multilabel to the branches below,
    # which is how a 3-class target silently scored as if it were multilabel.
    if y_test.ndim > 1 and y_test.shape[1] == 1:
        y_test = y_test.ravel()

    # In multioutput case, the prediction are in shape (n_class, n_samples, 2), and we want (n_samples, n_class) for positive label
    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, 1].T

    if y_test.ndim == 1:
        finite = y_test[~np.isnan(y_test)] if np.issubdtype(y_test.dtype, np.floating) else y_test
        classes = np.unique(finite)
        if len(classes) > 2:
            # Multiclass: One-vs-Rest needs the full (n_samples, n_classes)
            # probability matrix. Slicing column 1 (the binary convention below)
            # would silently score a single arbitrary class instead.
            return roc_auc_score(
                y_test, y_pred, average='macro', multi_class='ovr', labels=classes
            )
        # Binary: predict_proba gives (n_samples, 2); AUROC wants the positive column.
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        return roc_auc_score(y_test, y_pred, average='macro')

    return roc_auc_score(y_test, y_pred, average='macro', multi_class='ovr')


def _make_auroc_scorer(y: np.ndarray):
    labels = np.unique(y)
    if len(labels) > 2:
        log.info(f"Using multiclass OvR AUROC scorer over {len(labels)} classes")

        def multiclass_auroc(y_true, y_score):
            # labels= pins the expected class order to the full training set, so
            # a CV fold whose split happens to miss a class does not fail with
            # "Number of classes in y_true not equal to the number of columns".
            return roc_auc_score(
                y_true, y_score, multi_class="ovr", average="macro", labels=labels
            )

        return make_scorer(multiclass_auroc, response_method="predict_proba")

    # Binary: sklearn's built-in scorer takes the positive-class column of
    # predict_proba. Falling off the end here returned None, which makes
    # GridSearchCV silently select on the estimator default (accuracy).
    return "roc_auc"


def get_fp_model(cfg):
    return {
        'ChemeleonFingerprint': ChemeleonFingerprint,
        'ECFPFingerprint': ECFPFingerprint,
    }[cfg.model.cls](**cfg.model.params)


def _select_heads(cfg, downstream_models: dict) -> dict:
    # A model config may pin its own subset of heads (e.g. a transformer
    # embedder that only gets RF and LR); otherwise the global list applies.
    heads = cfg.model.get("heads") or cfg.heads
    unknown = set(heads) - set(downstream_models)
    if unknown:
        raise ValueError(
            f"Unknown downstream head(s) {sorted(unknown)}; "
            f"available: {sorted(downstream_models)}"
        )
    return {name: downstream_models[name] for name in heads}


def _embedder_name(cfg) -> str:
    # The config group choice ("chemeleon", "ecfp_count") is more specific than
    # cfg.model.cls, which cannot tell two ECFP variants apart.
    try:
        return HydraConfig.get().runtime.choices["model"]
    except Exception:
        return cfg.model.cls


def append_to_csv(row: dict, cfg):
    lock = FileLock(cfg.lock_file)
    df = pd.DataFrame([row])
    with lock:
        if os.path.exists(cfg.result_file):
            df.to_csv(cfg.result_file, mode='a', header=False, index=False)
        else:
            df.to_csv(cfg.result_file, index=False)


@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def main(cfg):
    # 1. Load dataset
    print(f"Loading dataset from {cfg.output_dir}/{cfg.dataset.name}_split.csv")
    data_filename = os.path.join(cfg.output_dir, cfg.dataset.name + "_split.csv")
    df = pd.read_csv(data_filename)
    fp_model = get_fp_model(cfg)
    embedder = _embedder_name(cfg)

    # 2. Embed dataset
    print(f"Embedding dataset with {cfg.model.cls}...")
    embed_start_time = time()
    X = fp_model.transform(df["SMILES"].tolist())
    embed_end_time = time()
    downstream_models = _select_heads(
        cfg, get_clf_models(no_output=1, embeddings_dtype=X.dtype)
    )

    for n_fold in tqdm(df['fold'].unique().tolist(), desc="Evaluating fold", disable=cfg.disable_tqdm):
        train_idx = df['fold'] != n_fold
        test_idx = df['fold'] == n_fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = df.loc[train_idx, 'label'], df.loc[test_idx, 'label']

        # 3. Evaluate GridSearchCV across all downstream models
        for model_name, model in tqdm(downstream_models.items(), desc="Evaluating model", leave=False, disable=cfg.disable_tqdm):
            start = time()
            cv = GridSearchCV(
                model["model"],
                model["params"],
                scoring=_make_auroc_scorer(y_train),
                cv=cfg.cv_folds,
                n_jobs=cfg.n_jobs,
            )
            cv.fit(X_train, y_train)
            best_model = cv.best_estimator_
            y_pred = best_model.predict_proba(X_test)
            end = time()

            # 4. Compute AUROC
            auroc = get_skfp_roc_auc(y_pred, y_test)
            append_to_csv({
                "dataset": cfg.dataset.name,
                "embedder": embedder,
                "fold": n_fold,
                "model": model_name,
                "auroc": auroc,
                "train_time": round(end - start, 3),
                "embed_time": round(embed_end_time - embed_start_time, 3),
            }, cfg)


if __name__ == "__main__":
    main()
