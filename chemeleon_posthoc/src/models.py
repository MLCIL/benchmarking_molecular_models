import numpy as np
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from skfp.distances import tanimoto_count_distance

from src.tanimoto_knn import BulkTanimotoKNeighborsClassifier

RANDOM_STATE = 42


RF_CLF = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["entropy"],
}

RF_REG = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["squared_error"],
}

RIDGE__MULTIOUTPUT_CLF = {
    "clf__estimator__C": 1 / np.logspace(-2, 3, 10),
    "clf__estimator__penalty": ["l2"],
    "clf__estimator__solver": ["lbfgs"],
    "clf__estimator__max_iter": [5000],
}

RIDGE_CLF = {
    "clf__C": 1 / np.logspace(-2, 3, 10),
    "clf__l1_ratio": [0.0],
    "clf__solver": ["lbfgs"],
    "clf__max_iter": [5000],
}


RIDGE_REG = {
    "clf__alpha": np.logspace(-2, 3, 10),
    "clf__max_iter": [5000],
    "clf__solver": ["lbfgs"],
}

KNN_CLF = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}

KNN_REG = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}


AVAILABLE_HEADS = ["rf", "ridge", "knn"]


def get_knn_distance(embeddings_dtype):
    if np.issubdtype(embeddings_dtype, np.integer):
        return tanimoto_count_distance
    elif np.issubdtype(embeddings_dtype, np.floating):
        return "cosine"
    else:
        raise ValueError(f"Unsupported embeddings dtype: {embeddings_dtype}. Expected integer or floating point type.")


def _knn_pipeline(embeddings_dtype):
    """k-NN pipeline for one embedding dtype.

    Integer embeddings (count fingerprints) use Tanimoto on the raw counts,
    vectorised as a matmul in src.tanimoto_knn. No StandardScaler here:
    Tanimoto is defined for non-negative vectors, and mean-centring the counts
    made the "distance" go negative -- it was no longer Tanimoto.

    Float embeddings keep StandardScaler + cosine.
    """
    if np.issubdtype(embeddings_dtype, np.integer):
        return Pipeline(
            [
                ("clf", BulkTanimotoKNeighborsClassifier(
                    n_jobs=-1, algorithm="brute",
                    metric=get_knn_distance(embeddings_dtype))),
            ]
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_jobs=-1, metric=get_knn_distance(embeddings_dtype))),
        ]
    )


def get_clf_models(no_output: int, embeddings_dtype):
    if no_output == 1:
        lr_clf = LogisticRegression(random_state=RANDOM_STATE)
        lr_params = RIDGE_CLF
    else:
        lr_clf = MultiOutputClassifier(LogisticRegression(random_state=RANDOM_STATE))
        lr_params = RIDGE__MULTIOUTPUT_CLF
    
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE))]),
            "params": RF_CLF.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", lr_clf),
                ]
            ),
            "params": lr_params.copy(),
        },
        "knn": {
            "model": _knn_pipeline(embeddings_dtype),
            "params": KNN_CLF.copy(),
        }
    }
