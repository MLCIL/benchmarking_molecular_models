"""One function per figure of the article, written as PNG to <out>/figures/updated/ (the manuscripts' paths).

Main text: workflow, bbt_cdd, cross_winrate, dataset_winrate, posthoc_forest, posthoc_simplex, compute_times.
Supplement: bbt_posterior_strong, nu_prior.
"""

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from . import config as C  # noqa: E402
from .analysis import Analysis  # noqa: E402
from .latex import save_png  # noqa: E402

log = logging.getLogger("published_figures")

POSTHOC_CV = "Scaffold 5-fold CV"  # evaluation scheme in the t-test plot titles


@contextmanager
def figure_style(seaborn_style: str | None = None):
    """Every figure starts from matplotlib's defaults (or a seaborn style), whatever was drawn before it."""
    with plt.rc_context():
        matplotlib.rc_file_defaults()
        if seaborn_style:
            sns.set_style(style=seaborn_style)
        yield


def rho_text(rho_label: str) -> str:
    return f"{rho_label} (Nadeau-Bengio)" if rho_label == C.POSTHOC_MAIN_RHO else rho_label


# --------------------------------------------------------------------------- main text


def workflow(a: Analysis, out: Path) -> None:
    """Figure workflow_fig: the evaluation workflow, a static drawing (not generated from results)."""
    shutil.copy2(C.STATIC / "workflow.png", out / "figures" / "workflow.png")
    log.info("Wrote %s", out / "figures" / "workflow.png")


def bbt_cdd(a: Analysis, out: Path) -> None:
    """Figure fig:bbt_cdd: critical difference diagram, models by posterior mean beta, bars join groups of
    pairwise practically equivalent models (P(pi in ROPE) >= 0.95)."""
    with figure_style():
        fig, ax = plt.subplots(figsize=(C.A4_TEXT_WIDTH, 3.0))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        a.bbt.plot_cdd_diagram(rope_value=C.BBT_ROPE, ax=ax, connector_kwargs={"color": "0.65", "linewidth": 0.6},
                               interpretation_note=f"A bar joins models with P(π ∈ [{C.BBT_ROPE[0]:g}, "
                                                   f"{C.BBT_ROPE[1]:g}]) ≥ 0.95")
        save_png(fig, out / C.FIGURES / "bbt_cdd")


def cross_winrate(a: Analysis, out: Path) -> None:
    """Figure fig:cross_win_rate: net win rate (wins - losses) / 25 of the row model against the column model."""
    win, order = a.net_win_rate, a.order
    win.to_csv(out / C.TABLES / "fig_cross_winrate.csv")
    with figure_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(20, 15))
        cmap = LinearSegmentedColormap.from_list("custom_diverging", ["red", "white", "green"])
        sns.heatmap(win, cmap=cmap, vmin=-1, vmax=1, linewidths=0.1, square=True, annot=True,
                    fmt=".0%", annot_kws={"size": 8}, cbar=False, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks(np.arange(len(order)) + 0.5, labels=order, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(order)) + 0.5, labels=order, rotation=0)
        fig.tight_layout()
        save_png(fig, out / C.FIGURES / "cross_winrate", dpi=400, pad_inches=0.1)


def dataset_winrate(a: Analysis, out: Path) -> None:
    """Figure fig:dataset_winrate: per dataset, the share of models exceeding ECFP's AUROC by more than 0.01."""
    s = a.dataset_winrate
    s.rename(columns={"mean": "share_models_beating_ecfp", "sum": "n_models_beating_ecfp"}).to_csv(
        out / C.TABLES / "fig_dataset_winrate.csv", index=False)
    with figure_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(18, 7))
        labels = s["dataset_name"].str.removeprefix("MoleculeNet_").str.replace("_", " ")
        sns.barplot(s, x="dataset_name", y="mean", ax=ax)
        ax.set_xticks(ax.get_xticks(), labels=labels, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Percentage of models outperforming ECFP")
        ax.set_xlabel(None)
        save_png(fig, out / C.FIGURES / "dataset_winrate", pad_inches=0.1)


def posthoc_forest(a: Analysis, out: Path) -> None:
    """Figure fig:posthoc_forest: per-dataset posterior of mu_i (CheMeleon - ECFP) at rho = 0, raw mean differences."""
    rho = C.POSTHOC_FOREST_RHO
    test = a.ttests[("corani", rho)]
    a.per_dataset(rho).to_csv(out / C.TABLES / f"fig_scaffold_rho_{rho}_forest.csv", index=False)
    with figure_style():
        fig, ax = plt.subplots(figsize=(C.A4_TEXT_WIDTH, 1.2 + 0.3 * len(C.POSTHOC_DATASETS)))
        test.plot(kind="forest", ax=ax, order=list(C.POSTHOC_DATASETS)[::-1])  # first dataset at the top
        ticks = ax.get_yticklabels()
        ax.set_yticks(ax.get_yticks(), labels=[C.POSTHOC_DATASETS[t.get_text()] for t in ticks],
                      fontsize=ticks[0].get_fontsize())
        ax.set_title(f"{ax.get_title()}\n{POSTHOC_CV}, $\\rho$ = {rho_text(rho)}", fontsize=9)
        save_png(fig, out / C.FIGURES / f"scaffold_rho_{rho}_forest")


def posthoc_simplex(a: Analysis, out: Path) -> None:
    """Figure fig:posthoc_simplex (panels rho = 0 and 0.2): posterior predictive simplex of the next dataset.

    The colour scale (draws per hexagon) is shared by the fits of all six rho, so panels are comparable.
    """
    from btk.tests.ttest.plots import plot_hierarchical_simplex

    left, right = C.POSTHOC_MODELS.values()
    with figure_style():
        panels = []
        for label in C.POSTHOC_RHOS:
            test = a.ttests[("corani", label)]
            fig, ax = plt.subplots(figsize=(C.A4_TEXT_WIDTH / 2, C.A4_TEXT_WIDTH / 2 * 0.9))
            plot_hierarchical_simplex(*test._next_dataset_triple((-C.TIE, C.TIE)), left_model=left,
                                      right_model=right, rope_band=(-C.TIE, C.TIE), ax=ax, gridsize=40,
                                      colorbar=False, norm=Normalize())
            ax.set_title(f"{POSTHOC_CV}, $\\rho$ = {rho_text(label)}", fontsize=9, pad=30)
            panels.append((label, fig, ax, ax.collections[0]))
        norm = Normalize(vmin=1, vmax=max(h.get_array().max() for *_, h in panels))
        for label, fig, ax, hexes in panels:
            if label not in C.POSTHOC_SIMPLEX_RHOS:
                plt.close(fig)
                continue
            hexes.set_norm(norm)
            cbar = fig.colorbar(hexes, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label("Posterior draws per cell (linear scale)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            save_png(fig, out / C.FIGURES / f"scaffold_rho_{label}_simplex")


def compute_times(a: Analysis, out: Path) -> None:
    """Figure fig:compute_times: mean time (± std over batches) to embed 10,000 MUV molecules, log scale."""
    best = a.runtime
    colors, names = {"cpu": "#2a78d6", "cuda": "#eb6834"}, {"cpu": "CPU", "cuda": "GPU (CUDA)"}
    with figure_style():
        fig, ax = plt.subplots(figsize=(C.A4_TEXT_WIDTH, 3.4))
        x = np.arange(len(best))
        for dev, color in colors.items():
            m = (best["device"] == dev).to_numpy()
            ax.bar(x[m], best["mean_runtime"][m], width=0.7, color=color, label=names[dev],
                   yerr=best["std_runtime"][m], capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "#444"},
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(best["family"], rotation=60, ha="right", rotation_mode="anchor", fontsize=7.5)
        ax.set_xlim(-0.6, len(best) - 0.4)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylabel("Time per 10,000 molecules (s)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", which="major", color="#ddd", linewidth=0.6, zorder=0)
        ax.grid(axis="y", which="minor", color="#eee", linewidth=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Device", fontsize=7.5, title_fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout()
        save_png(fig, out / C.FIGURES / "muv_compute_times", tight=False)


# --------------------------------------------------------------------------- supplement


def bbt_posterior_strong(a: Analysis, out: Path) -> None:
    """Figure sup:fig:bbt_strong_interpreatation: posterior mean of pi = P(model > ECFP) with its 89% HDI and the
    regions of the strong interpretation, models by posterior mean beta."""
    with figure_style():
        fig, ax = plt.subplots(figsize=(C.A4_TEXT_WIDTH, 0.8 + 0.2 * len(a.order)))
        a.bbt.plot_strong_posterior(control_model="ECFP", orientation="vertical", ax=ax)
        save_png(fig, out / C.FIGURES / "bbt_posterior_strong")


def nu_prior(a: Analysis, out: Path) -> None:
    """Figure sup:fig:nu_prior: prior (dotted) and posterior (solid) of nu at rho = 0.2, main analysis
    (hierarchical Gamma prior) and Juárez & Steel Gamma(2, 0.1)."""
    from scipy import stats as st

    rng = np.random.default_rng(C.SEED)
    main_prior = rng.gamma(rng.uniform(0.5, 5.0, 200_000), 1 / rng.uniform(0.05, 0.15, 200_000))
    grid = np.linspace(0, 150, 600)
    with figure_style():
        fig, ax = plt.subplots(figsize=(C.A4_TEXT_WIDTH, 3.0))
        ax.hist(main_prior[main_prior < 150], bins=150, density=True, histtype="step", color="#1f77b4",
                linestyle=":", label="prior, main analysis")
        ax.plot(grid, st.gamma.pdf(grid, C.JS_SHAPE, scale=1 / C.JS_RATE), color="#d62728", linestyle=":",
                label="prior, Juárez & Steel Gamma(2, 0.1)")
        for prior, color in [("corani", "#1f77b4"), ("juarez_steel", "#d62728")]:
            nu = a.ttests[(prior, C.POSTHOC_MAIN_RHO)]._idata.posterior["nu"].values.ravel()
            ax.hist(nu[nu < 150], bins=150, density=True, histtype="step", color=color, linewidth=1.5,
                    label=f"posterior, {'main analysis' if prior == 'corani' else 'Juárez & Steel'}")
        ax.set_xlim(0, 150)
        ax.set_xlabel(r"$\nu$ (degrees of freedom of the dataset-level Student $t$)")
        ax.set_ylabel("density")
        ax.set_title(rf"{POSTHOC_CV}, $\rho$ = {C.POSTHOC_MAIN_RHO}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        save_png(fig, out / C.FIGURES / "juarez_steel_prior", tight=False)


MAIN = [workflow, bbt_cdd, cross_winrate, dataset_winrate, posthoc_forest, posthoc_simplex, compute_times]
SUPPLEMENT = [bbt_posterior_strong, nu_prior]
