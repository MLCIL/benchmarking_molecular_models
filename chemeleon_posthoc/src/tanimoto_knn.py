"""Vectorised Tanimoto neighbours for the k-NN head.

Port of ``src/eval/supervised/tanimoto_knn.py`` from the main repo.

Count fingerprints are integer dtype, and handing sklearn
``skfp.distances.tanimoto_count_distance`` as a callable metric makes it
evaluate the distance once per pair of molecules in Python -- 25-50 min per CV
fold on the mid-size posthoc datasets, hours on NICEATM.

``tanimoto_count_similarity`` is

    sim(a, b) = a . b / (||a||^2 + ||b||^2 - a . b)

so a whole distance block is one BLAS matmul. These subclasses keep sklearn's
neighbour selection and voting and only swap the distance computation, by
overriding ``kneighbors`` (``fit`` stays free; distances are computed only for
the rows actually queried).
"""

from __future__ import annotations

import numpy as np

from sklearn.neighbors import KNeighborsClassifier


#: Query rows per block. Caps the distance block at chunk x n_train.
CHUNK_SIZE = 2048


class BulkTanimotoKNeighborsClassifier(KNeighborsClassifier):
    """KNeighborsClassifier with matmul-based Tanimoto count distance."""

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        fit_X = np.asarray(self._fit_X, dtype=np.float64)
        n_fit = fit_X.shape[0]

        # sklearn's contract: X=None means "the training data", with each point
        # excluded from its own neighbour list.
        drop_self = X is None
        query = fit_X if drop_self else np.asarray(X, dtype=np.float64)

        k = min(n_neighbors + (1 if drop_self else 0), n_fit)
        n_queries = query.shape[0]
        dists = np.empty((n_queries, k), dtype=np.float64)
        inds = np.empty((n_queries, k), dtype=np.int64)

        sq_fit = np.einsum("ij,ij->i", fit_X, fit_X)

        for start in range(0, n_queries, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, n_queries)
            block = query[start:stop]
            dot = block @ fit_X.T
            sq_block = np.einsum("ij,ij->i", block, block)
            union = sq_block[:, None] + sq_fit[None, :] - dot
            # all-zero pairs are identical by skfp's convention (similarity 1)
            sim = np.ones_like(dot)
            np.divide(dot, union, out=sim, where=union >= 1e-8)
            D = 1.0 - sim
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            d = np.take_along_axis(D, part, axis=1)
            order = np.argsort(d, axis=1, kind="stable")
            inds[start:stop] = np.take_along_axis(part, order, axis=1)
            dists[start:stop] = np.take_along_axis(d, order, axis=1)

        if drop_self:
            # Drop the self match column-wise rather than assuming it sorted
            # first: duplicate molecules tie with it at distance 0.
            keep = inds != np.arange(n_queries)[:, None]
            keep[np.cumsum(keep, axis=1) > n_neighbors] = False
            inds = inds[keep][: n_queries * n_neighbors].reshape(n_queries, n_neighbors)
            dists = dists[keep][: n_queries * n_neighbors].reshape(n_queries, n_neighbors)

        return (dists, inds) if return_distance else inds
