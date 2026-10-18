import numpy as np
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import  Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from skfp.distances import tanimoto_count_distance

from .tanimoto_knn import BULK_KNN


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
    "clf__penalty": ["l2"],
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


def _knn_pipeline(estimator_cls, embeddings_dtype):
    """k-NN pipeline for one embedding dtype.

    Integer embeddings (the fingerprints) use Tanimoto, vectorised as a matmul
    in tanimoto_knn rather than handed to sklearn as a per-pair callable.

    No StandardScaler on this branch. Tanimoto is defined for non-negative count
    or binary vectors; mean-centring makes the inputs negative, at which point
    the formula no longer yields a distance in [0, 1] -- it went negative, which
    sklearn's callable path silently accepted. Scaling first was measuring
    something that was not Tanimoto.

    Float embeddings keep StandardScaler + cosine, which is a normal and
    well-defined combination.
    """
    if np.issubdtype(embeddings_dtype, np.integer):
        return Pipeline(
            [
                ("clf", BULK_KNN[estimator_cls](
                    n_jobs=-1, algorithm="brute",
                    metric=get_knn_distance(embeddings_dtype))),
            ]
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", estimator_cls(n_jobs=-1, metric=get_knn_distance(embeddings_dtype))),
        ]
    )



def get_clf_models(no_output: int, embeddings_dtype):
    if no_output == 1:
        lr_clf = LogisticRegression(n_jobs=-1)
        lr_params = RIDGE_CLF
    else:
        lr_clf = MultiOutputClassifier(LogisticRegression(n_jobs=-1))
        lr_params = RIDGE__MULTIOUTPUT_CLF
    
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestClassifier(n_jobs=-1))]),
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
            "model": _knn_pipeline(KNeighborsClassifier, embeddings_dtype),
            "params": KNN_CLF.copy(),
        }
    }


def get_reg_models(embeddings_dtype):
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestRegressor(n_jobs=-1))]),
            "params": RF_REG.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", Ridge()),
                ]
            ),
            "params": RIDGE_REG.copy(),
        },
        "knn": {
            "model": _knn_pipeline(KNeighborsRegressor, embeddings_dtype),
            "params": KNN_REG.copy(),
        }
    }
