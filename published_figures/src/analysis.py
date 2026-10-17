"""Every computation the article's objects share, computed once and on demand.

`Analysis` holds the benchmark results, the two Bayesian Bradley-Terry fits (all 53 variants, then the 28
representatives) and the 12 hierarchical t-test fits (6 values of rho x 2 priors on nu). The figures, tables and
numbers modules only read from it.

Protocol: per dataset and variant the best classification head (max test AUROC over kNN, RF, LR); each model is
represented by its variant with the highest posterior mean beta in the BBT fit on all variants (fingerprints: count
variant, fixed by design); all comparisons come from a BBT refit on the representatives; AUROC differences of at
most 0.01 are ties.
"""

import copy
import hashlib
import logging
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data

log = logging.getLogger("published_figures")


# --------------------------------------------------------------------------- Bayesian Bradley-Terry


def fit_bbt(wide: pd.DataFrame, cores: int, draws: int, tune: int, target_accept: float):
    """BBT with the add rule for ties (|AUROC difference| <= 0.01), 4 chains."""
    from btk import BBTTest

    model = BBTTest(absolute_tie_threshold=C.TIE, tie_solver="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # convergence is checked in numbers.bbt_convergence
        model.fit(wide.reset_index(), dataset_col="dataset", draws=draws, tune=tune, chains=4, cores=cores,
                  target_accept=target_accept, random_seed=C.SEED, progressbar=False)
    return model


def interpretation_decision(raw: pd.Series, baseline_left: pd.Series) -> np.ndarray:
    """Library interpretation codes ('=', '?', '>' = left better) -> decision of the other model vs the baseline."""
    return np.select([raw == "=", raw == "?", (raw == ">") & baseline_left],
                     ["Practically equivalent", "Unknown", "Worse"], default="Better")


def bbt_vs_baseline(model, rope: tuple[float, float], baseline: str = "ECFP") -> pd.DataFrame:
    """Every model vs the baseline: posterior mean of pi, 89% HDI, P(above 0.5), P(in ROPE), decisions.

    Weak interpretation (main text): P(in ROPE) >= 0.95 -> equivalent, else P(pi > 0.5) >= 0.95 -> better/worse.
    Strong interpretation (supplement): posterior mean > 0.70 -> better, in [0.45, 0.55] -> equivalent, < 0.30 ->
    worse, otherwise unknown, split by the side of the equivalence region. Rows by P(model > baseline), descending.
    """
    t = model.posterior_table(
        rope_value=rope, control_model=baseline,
        columns=["left_model", "right_model", "mean", "hdi_low", "hdi_high", "delta", "above_50", "in_rope",
                 "weak_interpretation_raw", "strong_interpretation_raw"],
        round_ndigits=None,
    )
    baseline_left = t["left_model"] == baseline
    t["other"] = np.where(baseline_left, t["right_model"], t["left_model"])
    t["p_other"] = np.where(baseline_left, 1 - t["mean"], t["mean"])  # P(other > baseline)
    t["p_other_hdi_low"] = np.where(baseline_left, 1 - t["hdi_high"], t["hdi_low"])
    t["p_other_hdi_high"] = np.where(baseline_left, 1 - t["hdi_low"], t["hdi_high"])
    t["decision"] = interpretation_decision(t["weak_interpretation_raw"], baseline_left)
    strong = interpretation_decision(t["strong_interpretation_raw"], baseline_left)
    t["decision_strong"] = np.where(strong == "Unknown",
                                    np.where(baseline_left, C.STRONG_DECISIONS[3], C.STRONG_DECISIONS[1]), strong)
    return t.sort_values("p_other", ascending=False).reset_index(drop=True)


def subset_bbt(model, labels: dict[str, str]):
    """View of a fitted BBTTest restricted to some of its models and relabelled (fitted name -> new name).

    Keeps their columns of the beta posterior as they are (no refit), so the view reads the posterior of the
    larger fit.
    """
    import arviz as az

    view = copy.copy(model)
    post = model._idata.posterior
    idx = [model._algorithms.index(name) for name in labels]
    view._idata = view._fit_posterior = az.InferenceData(posterior=post.isel({post["beta"].dims[-1]: idx}))
    view._algorithms = list(labels.values())
    return view


# --------------------------------------------------------------------------- hierarchical t-test


def data_hash(wide: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(wide, index=True).values.tobytes()).hexdigest()


def _cache_key(wide: pd.DataFrame, prior: str, rho) -> dict:
    return {"data_hash": data_hash(wide), "prior": prior, "rho": rho, "seed": C.SEED, **C.POSTHOC_SAMPLING}


def fit_ttest(wide: pd.DataFrame, prior: str, rho, cores: int, path: Path) -> None:
    """Fit one hierarchical t-test (CheMeleon - ECFP, ROPE 0.01) and pickle it with the key it was fitted for.

    Runs in a worker process. prior = "corani" is the library's hierarchical prior on nu (main analysis);
    "juarez_steel" is nu ~ Gamma(2, 0.1), implemented by the model builder but not exposed by the public API.
    """
    import matplotlib

    matplotlib.use("Agg")
    from btk import HierarchicalTTest

    test = HierarchicalTTest(rope=C.TIE, parametrization="non_centered")
    if prior == "juarez_steel":
        test._dof_prior = "juarez_steel"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # convergence is checked in numbers.nu_prior_sensitivity
        test.fit(wide, dataset_col="dataset", fold_col="fold", rho=rho, **C.POSTHOC_SAMPLING, cores=cores,
                 random_seed=C.SEED, progressbar=False)
    with open(path, "wb") as f:
        pickle.dump({"test": test, "key": _cache_key(wide, prior, rho)}, f)


def ttest_stats(test, prior: str, rho_label: str) -> dict:
    """Decision table of one fit plus the posterior of nu and the sampler diagnostics."""
    row = test.decision_table(columns=test.ALL_TTEST_COLUMNS, round_ndigits=None).iloc[0].to_dict()
    nu = test._idata.posterior["nu"].values.ravel()
    diag = test.diagnostics()
    return {
        "prior": prior, "rho_label": rho_label, **row,
        "nu_median": np.median(nu), "nu_q05": np.quantile(nu, 0.05), "nu_q95": np.quantile(nu, 0.95),
        # Monte-Carlo error of P(equivalent), conservatively from the smallest bulk ESS
        "mcse_p_rope": float(np.sqrt(max(row["p_rope"] * (1 - row["p_rope"]), 1e-12) / diag["min_ess_bulk"])),
        **diag,
    }


# --------------------------------------------------------------------------- shared state


class Analysis:
    """Lazily computed state shared by all objects of the article."""

    def __init__(self, results: Path, cache_dir: Path, cores: int = 4, jobs: int = 12, refit: bool = False):
        self.results, self.cache_dir = results, cache_dir
        self.cores, self.jobs, self.refit = cores, jobs, refit

    # ----------------------------------------------------------------- benchmark results

    @cached_property
    def results_all(self) -> pd.DataFrame:
        """AUROC of all 53 variants x 25 datasets x 3 heads."""
        return data.load_results(self.results)

    @cached_property
    def best(self) -> pd.DataFrame:
        """Best head per variant and dataset."""
        return data.best_head(self.results_all)

    @cached_property
    def bbt_variants(self):
        """BBT on all 53 variants (best head per dataset), used to select the representatives."""
        wide = self.best.assign(variant=self.best["embedder"].map(lambda e: C.MODELS[e][1])).pivot(
            index="dataset", columns="variant", values="test_metric")
        log.info("Fitting the BBT on %d variants", wide.shape[1])
        return fit_bbt(wide, self.cores, **C.BBT_VARIANT_SAMPLING)

    @cached_property
    def variant_beta(self) -> pd.Series:
        """Posterior mean beta of every variant (fit on all variants), descending."""
        return pd.Series(self.bbt_variants.beta_ranking, name="beta_mean").sort_values(ascending=False)

    @cached_property
    def family_representatives(self) -> dict[str, str]:
        """Model -> variant with the highest posterior mean beta, or the variant fixed by design."""
        embedder = {C.MODELS[e][1]: e for e in C.MODELS}
        by_family = self.variant_beta.groupby(lambda v: C.MODELS[embedder[v]][0]).idxmax()
        reps = {f: embedder[v] for f, v in by_family.items()}
        for f, e in C.FIXED_REPRESENTATIVES.items():
            if reps[f] != e:
                log.warning("BBT picks %s for %s, overridden by the fixed %s", reps[f], f, e)
                reps[f] = e
        return reps

    @cached_property
    def reps(self) -> dict[str, str]:
        """Name of a model in the model-level tables and figures -> its representative variant (embedder)."""
        return {C.REP_LABELS.get(e, f): e for f, e in self.family_representatives.items()}

    @cached_property
    def cand(self) -> pd.DataFrame:
        """Best-head AUROC of the 28 representatives, with their model name in `family`."""
        rep_to_family = {e: f for f, e in self.reps.items()}
        cand = self.best[self.best["embedder"].isin(rep_to_family)].copy()
        cand["family"] = cand["embedder"].map(rep_to_family)
        return cand

    @cached_property
    def mean_rank(self) -> pd.DataFrame:
        """Mean rank (average rank among the representatives on each dataset) and mean AUROC, by mean rank."""
        c = self.cand.copy()
        c["rank"] = c.groupby("dataset")["test_metric"].rank(ascending=False, method="average")
        t = c.groupby("family").agg(mean_rank=("rank", "mean"), mean_auroc=("test_metric", "mean"))
        t = t.sort_values("mean_rank").reset_index()
        t["embedder"] = t["family"].map(self.reps)
        return t

    @cached_property
    def order(self) -> list[str]:
        """Models by mean rank (Table tab:mean_rank, the cross win rate figure, the BBT refit)."""
        return self.mean_rank["family"].tolist()

    @cached_property
    def by_dataset(self) -> pd.DataFrame:
        """Datasets x models (representatives), best-head AUROC."""
        return self.cand.pivot(index="dataset", columns="family", values="test_metric")

    @cached_property
    def bbt(self):
        """BBT refit on the 28 representatives: every comparison of the Results section."""
        wide = self.by_dataset[self.order]
        log.info("Fitting the BBT on the %d representatives", wide.shape[1])
        return fit_bbt(wide, self.cores, **C.BBT_SAMPLING)

    @cached_property
    def bbt_vs_ecfp(self) -> pd.DataFrame:
        """Refit: every model vs ECFP at the main ROPE (0.35, 0.65)."""
        return bbt_vs_baseline(self.bbt, C.BBT_ROPE)

    @cached_property
    def bbt_variants_vs_ecfp(self) -> pd.DataFrame:
        """The same comparisons read from the posterior of the fit on all 53 variants."""
        view = subset_bbt(self.bbt_variants, {C.MODELS[self.reps[f]][1]: f for f in self.order})
        return bbt_vs_baseline(view, C.BBT_ROPE)

    @cached_property
    def full_comparison(self) -> pd.DataFrame:
        """All variants x heads (and the best head): mean rank among all variants and mean AUROC."""
        df = self.results_all
        r = pd.concat([df[["embedder", "dataset", "model", "test_metric"]], self.best.assign(model="best")])
        r["rank"] = r.groupby(["dataset", "model"])["test_metric"].rank(ascending=False, method="average")
        t = r.groupby(["embedder", "model"]).agg(rank=("rank", "mean"), auroc=("test_metric", "mean")).unstack("model")
        out = pd.DataFrame({"embedder": t.index, "variant": [C.MODELS[e][1] for e in t.index]})
        for metric in ["rank", "auroc"]:
            for m in ["best", *C.HEADS]:
                out[f"{metric}_{C.HEADS.get(m, m)}"] = t[(metric, m)].values
        return out.sort_values("rank_best").reset_index(drop=True)

    @cached_property
    def near_wins(self) -> pd.DataFrame:
        """Datasets on which a model won or was within 0.01 AUROC of the winner."""
        win = self.cand.groupby("dataset")["test_metric"].transform("max")
        c = self.cand.assign(near_win=(win - self.cand["test_metric"]) <= C.TIE)
        t = c.groupby("family")["near_win"].sum().astype(int).reset_index(name="n_datasets")
        return t.sort_values(["n_datasets", "family"], ascending=[False, True]).reset_index(drop=True)

    @cached_property
    def net_win_rate(self) -> pd.DataFrame:
        """Row model vs column model: (wins - losses) / 25 datasets, ties at |difference| <= 0.01; mean rank order."""
        vals = self.by_dataset[self.order].values
        diff = vals[:, :, None] - vals[:, None, :]
        score = np.where(np.abs(diff) <= C.TIE, 0, np.sign(diff))
        return pd.DataFrame(score.mean(axis=0), index=self.order, columns=self.order)

    @cached_property
    def dataset_winrate(self) -> pd.DataFrame:
        """Per dataset: share and number of the other 27 models exceeding ECFP's AUROC by more than 0.01."""
        ecfp = self.cand[self.cand["embedder"] == C.BASELINE].set_index("dataset")["test_metric"]
        other = self.cand[self.cand["embedder"] != C.BASELINE].copy()
        other["won"] = (other["test_metric"] - other["dataset"].map(ecfp)) > C.TIE
        s = other.groupby("dataset")["won"].agg(["mean", "sum"]).reset_index()
        s = s.sort_values(["mean", "dataset"], ascending=[False, True])
        s["dataset_name"] = s["dataset"].replace(C.DATASET_NAMES)
        return s

    def head_to_head(self, model: str) -> tuple[int, int, int]:
        """Wins, ties (|difference| <= 0.01) and losses of a model against ECFP over the 25 datasets."""
        d = self.by_dataset[model] - self.by_dataset["ECFP"]
        return int((d > C.TIE).sum()), int((d.abs() <= C.TIE).sum()), int((d < -C.TIE).sum())

    @cached_property
    def variant_selection(self) -> pd.DataFrame:
        """Per model: variant with the highest mean (best-head) AUROC vs with the highest BBT strength."""
        mean_auc = self.best.groupby("embedder")["test_metric"].mean()
        v = pd.DataFrame({"embedder": mean_auc.index, "mean_auroc": mean_auc.values})
        v["family"] = v["embedder"].map(lambda e: C.MODELS[e][0])
        v["beta_mean"] = v["embedder"].map(lambda e: self.variant_beta[C.MODELS[e][1]])
        by_auc = v.loc[v.groupby("family")["mean_auroc"].idxmax()].set_index("family")["embedder"]
        by_bbt = v.loc[v.groupby("family")["beta_mean"].idxmax()].set_index("family")["embedder"]
        t = pd.DataFrame({"family": sorted(by_auc.index)})
        t["n_variants"] = t["family"].map(v.groupby("family").size())
        t["auroc_variant"] = t["family"].map(by_auc)
        t["bbt_variant"] = t["family"].map(by_bbt)
        t["same"] = t["auroc_variant"] == t["bbt_variant"]
        return t.sort_values(["same", "family"]).reset_index(drop=True)

    # ----------------------------------------------------------------- runtime

    @cached_property
    def runtime(self) -> pd.DataFrame:
        """Embedding time on MUV of every model's representative variant, slowest first.

        CLAMP is timed as the authors' checkpoint (config.RUNTIME_EMBEDDERS; same architecture as the retrained
        model). For MAT the last rows (8 full batches) drop batch 9, which holds one molecule whose conformer
        generation takes 73 s.
        """
        rt = data.load_runtime(self.results).rename(columns={"embedder": "timed_embedder"})
        best = self.mean_rank[["family", "embedder", "mean_rank"]].copy()
        best["timed_embedder"] = best["embedder"].map(lambda e: C.RUNTIME_EMBEDDERS.get(e, e))
        best = best.merge(rt, on="timed_embedder", how="left", validate="1:1")
        if best["mean_runtime"].isna().any():
            raise ValueError(f"No runtime for {best.loc[best['mean_runtime'].isna(), 'timed_embedder'].tolist()}")
        return best.sort_values("mean_runtime", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------------- external validation

    @cached_property
    def posthoc_folds(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-fold AUROC (dataset, fold, CheMeleon, ECFP) of the head with the best mean AUROC, and those heads."""
        df = data.load_posthoc(self.results)
        mean = df.groupby(["dataset", "embedder", "model"])["auroc"].mean().reset_index()
        heads = mean.loc[mean.groupby(["dataset", "embedder"])["auroc"].idxmax()].rename(columns={"auroc": "mean_auroc"})
        best = df.merge(heads[["dataset", "embedder", "model"]], on=["dataset", "embedder", "model"])
        wide = best.pivot(index=["dataset", "fold"], columns="embedder", values="auroc")
        wide = wide[list(C.POSTHOC_MODELS)].rename(columns=C.POSTHOC_MODELS).reset_index()
        return wide, heads

    @cached_property
    def posthoc_raw(self) -> pd.DataFrame:
        """Per external dataset (manuscript order): mean and std over folds (ddof=1) of both models and their difference."""
        wide, _ = self.posthoc_folds
        left, right = C.POSTHOC_MODELS.values()
        w = wide.assign(diff=wide[left] - wide[right])
        out = w.groupby("dataset")[[left, right, "diff"]].agg(["mean", "std"])
        out.columns = [f"{c}_{s}" for c, s in out.columns]
        return posthoc_ordered(out.reset_index())

    @cached_property
    def ttests(self) -> dict[tuple[str, str], object]:
        """(prior on nu, rho label) -> fitted hierarchical t-test; 12 fits, pickled in the cache directory.

        A cached fit is reused when it was fitted on the same fold scores with the same settings; `refit` forces
        new fits. The fits run `jobs` at a time, each sampling its 4 chains on `cores` processes.
        """
        wide, _ = self.posthoc_folds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys = [(prior, label) for prior in C.NU_PRIORS for label in C.POSTHOC_RHOS]
        path = {k: self.cache_dir / f"ttest_{k[0]}_rho_{k[1]}.pkl" for k in keys}

        def load(k):
            if self.refit or not path[k].exists():
                return None
            with open(path[k], "rb") as f:
                cached = pickle.load(f)
            return cached["test"] if cached["key"] == _cache_key(wide, k[0], C.POSTHOC_RHOS[k[1]]) else None

        tests = {k: load(k) for k in keys}
        todo = [k for k, t in tests.items() if t is None]
        if todo:
            log.info("Fitting %d hierarchical t-tests (%d at a time, %d cores each)", len(todo), self.jobs, self.cores)
            with ProcessPoolExecutor(max_workers=self.jobs, mp_context=get_context("spawn")) as pool:
                for fut in [pool.submit(fit_ttest, wide, k[0], C.POSTHOC_RHOS[k[1]], self.cores, path[k]) for k in todo]:
                    fut.result()
            tests.update({k: load(k) for k in todo})
        return tests

    @cached_property
    def ttest_stats(self) -> pd.DataFrame:
        """One row per prior on nu and rho: E[mu_0], 89% HDI, P(ECFP) / P(equivalent) / P(CheMeleon), nu, diagnostics."""
        return pd.DataFrame([ttest_stats(t, prior, label) for (prior, label), t in self.ttests.items()])

    @cached_property
    def posthoc_stats(self) -> pd.DataFrame:
        """Main analysis (hierarchical prior on nu), one row per rho."""
        return self.ttest_stats[self.ttest_stats["prior"] == "corani"].reset_index(drop=True)

    def per_dataset(self, rho_label: str, prior: str = "corani") -> pd.DataFrame:
        """Posterior of the per-dataset mean differences mu_i (shrunk mean, 89% HDI), manuscript order."""
        return posthoc_ordered(self.ttests[(prior, rho_label)].per_dataset_table(round_ndigits=None))


def posthoc_ordered(df: pd.DataFrame, col: str = "dataset") -> pd.DataFrame:
    """Rows in the order of tab:external_datasets, with the manuscript's dataset names in `dataset_name`."""
    if set(df[col]) != set(C.POSTHOC_DATASETS):
        raise ValueError(f"Datasets differ from POSTHOC_DATASETS: {sorted(set(df[col]) ^ set(C.POSTHOC_DATASETS))}")
    rank = {d: i for i, d in enumerate(C.POSTHOC_DATASETS)}
    out = df.sort_values(col, key=lambda c: c.map(rank)).reset_index(drop=True)
    out.insert(out.columns.get_loc(col) + 1, "dataset_name", out[col].map(C.POSTHOC_DATASETS))
    return out
