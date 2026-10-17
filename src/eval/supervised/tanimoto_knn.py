"""Vectorised Tanimoto neighbours for the k-NN head.

Fingerprint embeddings are integer dtype, so ``get_knn_distance`` hands sklearn
``skfp.distances.tanimoto_count_distance`` as a *callable*. sklearn has no
vectorised path for a callable metric and evaluates it once per pair of
molecules -- ~82k pairs/s, single threaded. That is ~44 min for one CV fit on
ogbg-molhiv and ~228 min on ogbg-molmuv, roughly 14 h for the six fingerprint
embedders across both datasets.

``skfp.distances.tanimoto_count_similarity`` is defined as

    sim(a, b) = a . b / (||a||^2 + ||b||^2 - a . b)

which is entirely dot products and squared norms, so a whole distance block is
one BLAS matmul. These subclasses keep sklearn's neighbour bookkeeping and
voting intact and swap only the distance computation, by overriding
``kneighbors``.

``skfp`` also ships ``bulk_tanimoto_count_distance``, which computes the same
values (agreement 1.1e-14) and would avoid reimplementing the formula. It is not
used here because its throughput collapses as the fitted set grows: against a
29,579-row block it manages 0.32M pairs/s versus 99-120M pairs/s for the matmul
below -- 307x slower, and only ~4x better than the per-pair callable it was
meant to replace. Measured on this host; re-check before swapping it back in.

Why override ``kneighbors`` rather than precompute a graph in the pipeline: a
``KNeighborsTransformer``-style step has its ``fit_transform`` called by
``Pipeline.fit``, which builds the full train-vs-train graph. The callable path
never did that -- it computes distances only when queried -- so a transformer
adds ``n_train^2`` pairs per fit on top of the ``n_val x n_train`` that is
actually needed. On a MUV fold that is 4.5G extra pairs per fit, ~80% of the
work. Overriding ``kneighbors`` keeps ``fit`` free, as before.

Only the distance *computation* changes. Neighbour selection and voting stay in
sklearn, so multi-output behaviour is untouched.
"""

from __future__ import annotations

import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


#: Query rows per bulk call. Caps the distance block at chunk x n_train instead
#: of materialising the whole thing (34 GB on MUV).
CHUNK_SIZE = 2048


class _BulkTanimotoNeighborsMixin:
    """Replace sklearn's per-pair metric loop with skfp's bulk Tanimoto."""

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        fit_X = np.asarray(self._fit_X, dtype=np.float64)
        n_fit = fit_X.shape[0]

        # sklearn's contract: querying with X=None means "the training data",
        # and each point is then excluded from its own neighbour list.
        drop_self = X is None
        query = fit_X if drop_self else np.asarray(X, dtype=np.float64)

        k = min(n_neighbors + (1 if drop_self else 0), n_fit)
        n_queries = query.shape[0]
        dists = np.empty((n_queries, k), dtype=np.float64)
        inds = np.empty((n_queries, k), dtype=np.int64)

        # ||a||^2 for the fitted rows, reused by every chunk.
        sq_fit = np.einsum("ij,ij->i", fit_X, fit_X)

        for start in range(0, n_queries, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, n_queries)
            block = query[start:stop]
            dot = block @ fit_X.T
            sq_block = np.einsum("ij,ij->i", block, block)
            D = 1.0 - dot / (sq_block[:, None] + sq_fit[None, :] - dot)
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            d = np.take_along_axis(D, part, axis=1)
            order = np.argsort(d, axis=1)
            inds[start:stop] = np.take_along_axis(part, order, axis=1)
            dists[start:stop] = np.take_along_axis(d, order, axis=1)

        if drop_self:
            # The self match is the nearest at distance 0; drop it column-wise
            # rather than assuming it sorted first, since duplicate molecules
            # tie with it at exactly 0.
            keep = inds != np.arange(n_queries)[:, None]
            # Guarantee exactly n_neighbors survive even when self did not
            # appear (possible only if k was clipped to n_fit).
            keep[np.cumsum(keep, axis=1) > n_neighbors] = False
            rows = np.repeat(np.arange(n_queries), n_neighbors)
            inds = inds[keep][: n_queries * n_neighbors].reshape(n_queries, n_neighbors)
            dists = dists[keep][: n_queries * n_neighbors].reshape(n_queries, n_neighbors)

        return (dists, inds) if return_distance else inds


class BulkTanimotoKNeighborsClassifier(_BulkTanimotoNeighborsMixin, KNeighborsClassifier):
    pass


class BulkTanimotoKNeighborsRegressor(_BulkTanimotoNeighborsMixin, KNeighborsRegressor):
    pass


BULK_KNN = {
    KNeighborsClassifier: BulkTanimotoKNeighborsClassifier,
    KNeighborsRegressor: BulkTanimotoKNeighborsRegressor,
}
