"""One function per group of numbers quoted in the text of the article.

Each function returns `Number`s: the claim, the value as printed in the article and the value computed from the
results in the same format. make_figures.py writes them to <out>/numbers.md and flags every disagreement, so a
change of the results or of the manuscript shows up there. The article values are those of main.tex /
supplementary.tex of 2026-10-17.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .analysis import Analysis
from .data import load_benchmark_datasets, load_external_labels, load_runtime


@dataclass
class Number:
    statement: str
    article: str  # as printed in the article
    computed: str  # from the results, in the article's format
    ok: bool | None = None  # agrees with the article; default: article == computed. None after __post_init__: n/a

    def __post_init__(self):
        if self.ok is None and self.computed is not None:
            self.ok = self.article == self.computed
        if self.ok is not None:
            self.ok = bool(self.ok)  # numpy booleans are not `False`


def and_join(names: list[str]) -> str:
    names = list(names)
    return "none" if not names else names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def main_name(model: str) -> str:
    return C.MAIN_TEXT_NAMES.get(model, model)


def short_name(dataset_name: str) -> str:
    """External dataset as the running text names it (NICEATM Acute Oral Toxicity -> NICEATM)."""
    return "NICEATM" if dataset_name.startswith("NICEATM") else dataset_name


def signed_pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:+.{digits}f}%"


# --------------------------------------------------------------------------- main text


def scope(a: Analysis) -> list[Number]:
    """Abstract and Methods: models, datasets, variants."""
    n_models = len(a.order)
    n_mn = sum(d.startswith("ogbg-") for d in C.DATASETS)
    n_fp = sum(f in C.FINGERPRINTS for f in a.order)
    classes = {d: load_external_labels(a.results, d).nunique() for d in C.POSTHOC_DATASETS}
    multi = [C.POSTHOC_DATASETS[d] for d, k in classes.items() if k == 3]
    return [
        Number("Models evaluated", "28", str(n_models)),
        Number("Classification datasets", "25", str(len(C.DATASETS))),
        Number("Datasets per benchmark", "7 from MoleculeNet and 18 from TDC",
               f"{n_mn} from MoleculeNet and {len(C.DATASETS) - n_mn} from TDC"),
        Number("Neural networks and fingerprints", "24 neural networks and 4 molecular fingerprints",
               f"{n_models - n_fp} neural networks and {n_fp} molecular fingerprints"),
        Number("Variants (BBT fit on all variants)", "53", str(a.results_all["embedder"].nunique())),
        Number("External validation datasets", "11", str(len(classes))),
        Number("External datasets: binary / three-class", "9 binary and 2 with three classes (DILIrank 2.0 and NICEATM)",
               f"{sum(k == 2 for k in classes.values())} binary and {len(multi)} with three classes "
               f"({and_join(sorted(map(short_name, multi)))})"),
    ]


def variant_selection(a: Analysis) -> list[Number]:
    """Methods, "Selection of model variants": representatives and the 53-variant posterior."""
    vs = a.variant_selection.set_index("family")
    fp_picks = [vs.loc[f, "bbt_variant"] for f in ["ECFP", "Atom Pair", "TT"]]
    weak, weak_all = a.bbt_vs_ecfp.set_index("other"), a.bbt_variants_vs_ecfp.set_index("other")
    differ = [m for m in a.order if m != "ECFP" and weak.loc[m, "decision"] != weak_all.loc[m, "decision"]]
    return [
        Number("BBT picks the count variant of every fingerprint (ECFP, Atom Pair, TT)", "count variants",
               "count variants" if all(e.endswith("_count") for e in fp_picks) else and_join(fp_picks)),
        Number("Decisions vs ECFP that differ between the 53-variant posterior and the refit (ROPE 0.35-0.65)",
               "Mol2vec", and_join(main_name(m) for m in differ)),
        Number("Mol2vec vs ECFP, P_in_ROPE, 53-variant posterior", "0.955", f"{weak_all.loc['Mol2Vec', 'in_rope']:.3f}"),
        Number("Mol2vec vs ECFP, P_in_ROPE, refit", "0.886", f"{weak.loc['Mol2Vec', 'in_rope']:.3f}"),
        Number("Mol2vec decision: 53-variant posterior / refit", "practically equivalent / worse",
               f"{weak_all.loc['Mol2Vec', 'decision'].lower()} / {weak.loc['Mol2Vec', 'decision'].lower()}"),
    ]


def rope_beta_bound(a: Analysis) -> list[Number]:
    """Methods, BBT: the ROPE on pi as a bound on beta_i - beta_j."""
    return [Number("ROPE [0.35, 0.65] <=> |beta_i - beta_j| <= logit(0.65)", "0.62",
                   f"{np.log(C.BBT_ROPE[1] / (1 - C.BBT_ROPE[1])):.2f}")]


def mean_rank(a: Analysis) -> list[Number]:
    """Results, Table tab:mean_rank."""
    t = a.mean_rank
    above = t.loc[t["mean_rank"] < t.set_index("family").loc["ECFP", "mean_rank"], "family"].tolist()
    return [
        Number("Models above ECFP in mean rank", "four", {4: "four"}.get(len(above), str(len(above)))),
        Number("... namely", "CheMeleon, R-MAT, MolBERT and ChemBERTa", and_join(above)),
        Number("Top model by mean rank / by mean AUROC", "CheMeleon / CheMeleon",
               f"{t.iloc[0]['family']} / {t.loc[t['mean_auroc'].idxmax(), 'family']}"),
    ]


def chemberta_mtr(a: Analysis) -> list[Number]:
    """Results and supplement "Additional detailed results": ChemBERTa MTR vs MLM (ranks among all variants)."""
    f = a.full_comparison.set_index("embedder")
    sizes, heads = ["5M", "10M", "77M"], ["best", *C.HEADS.values()]
    mtr, mlm = (lambda s: f.loc[f"ChemBERTa-{s}-MTR"]), (lambda s: f.loc[f"ChemBERTa-{s}-MLM"])
    every = all(mtr(s)[f"rank_{h}"] < mlm(s)[f"rank_{h}"] and mtr(s)[f"auroc_{h}"] > mlm(s)[f"auroc_{h}"]
                for s in sizes for h in heads)
    out = [Number("MTR beats MLM for every size and classifier (mean rank and mean AUROC)", "yes",
                  "yes" if every else "no")]
    for s, (rank, auc) in zip(sizes, [("23.70", "5.42%"), ("17.36", "3.54%"), ("24.30", "6.37%")]):
        out += [Number(f"{s}: MTR improves the mean rank by", rank, f"{mlm(s)['rank_best'] - mtr(s)['rank_best']:.2f}"),
                Number(f"{s}: MTR improves the mean AUROC by", auc, f"{mtr(s)['auroc_best'] - mlm(s)['auroc_best']:.2%}")]
    worst_mtr = max(mtr(s)["rank_best"] for s in sizes)
    best_mlm = min(mlm(s)["rank_best"] for s in sizes)
    out.append(Number("Every MTR variant ranks above every MLM variant", "yes", "yes" if worst_mtr < best_mlm else "no"))
    return out


def bbt_decisions(a: Analysis) -> list[Number]:
    """Results, "Bayesian testing results" (Table tab:bbt_decisions)."""
    bt = a.bbt_vs_ecfp
    by = lambda d: bt.loc[bt["decision"] == d]  # noqa: E731
    worse = by("Worse")
    hi, lo = worse.iloc[0], worse.iloc[-1]
    support = np.where(bt["decision"] == "Practically equivalent", bt["in_rope"], bt["above_50"])
    strong = bt.set_index("other")["decision_strong"]
    return [
        Number("Better than ECFP", "CheMeleon", and_join(by("Better")["other"])),
        Number("Practically equivalent to ECFP", "R-MAT, MolBERT, ChemBERTa, Mordred, CDDD, MAT, Atom Pair, MolE and "
               "MoLFormer", and_join(by("Practically equivalent")["other"])),
        Number("Inconclusive comparisons", "0", str(len(by("Unknown")))),
        Number("Worse than ECFP", "all other models",
               "all other models" if len(worse) == len(bt) - len(by("Better")) - len(by("Practically equivalent"))
               else and_join(worse["other"])),
        Number("Worse models: highest posterior mean P(model > ECFP)", "0.38 (Mol2vec)",
               f"{hi['p_other']:.2f} ({main_name(hi['other'])})"),
        Number("Worse models: lowest posterior mean P(model > ECFP)", "0.08 (SELFormer)",
               f"{lo['p_other']:.2f} ({main_name(lo['other'])})"),
        Number("Smallest posterior probability supporting a decision vs ECFP (P_in_ROPE or P_above_0.5)",
               "at least 99.5%", f"{support.min():.2%}", support.min() >= 0.995),
        Number("Strong interpretation: better than ECFP / CheMeleon", "none / inconclusive",
               f"{and_join(strong[strong == 'Better'].index)} / "
               f"{'inconclusive' if strong['CheMeleon'].startswith('Unknown') else strong['CheMeleon'].lower()}"),
    ]


def bbt_convergence(a: Analysis) -> list[Number]:
    """Methods, BBT: convergence of both fits is checked (R-hat <= 1.01, no divergences)."""
    out = []
    for name, model in [("53 variants", a.bbt_variants), ("28 representatives", a.bbt)]:
        d = model.diagnostics()
        out.append(Number(f"BBT fit on {name}: max R-hat, divergences, min bulk ESS", "converged",
                          f"{d['max_r_hat']:.2f}, {d['divergences']:.0f}, {d['min_ess_bulk']:.0f}",
                          d["max_r_hat"] <= 1.01 and d["divergences"] == 0))
    return out


def cdd(a: Analysis) -> list[Number]:
    """Results, Figure fig:bbt_cdd: practical equivalence is not transitive."""
    from .analysis import bbt_vs_baseline

    vs_chem = bbt_vs_baseline(a.bbt, C.BBT_ROPE, baseline="CheMeleon").set_index("other")["decision"]
    vs_ecfp = a.bbt_vs_ecfp.set_index("other")["decision"]
    word = {"Practically equivalent": "equivalent", "Better": "better", "Worse": "worse", "Unknown": "inconclusive"}
    flip = {"better": "worse", "worse": "better"}
    return [Number("CheMeleon vs R-MAT / R-MAT vs ECFP / CheMeleon vs ECFP", "equivalent / equivalent / better",
                   f"{flip.get(word[vs_chem['R-MAT']], word[vs_chem['R-MAT']])} / {word[vs_ecfp['R-MAT']]} / "
                   f"{word[vs_ecfp['CheMeleon']]}")]


def net_win_rate(a: Analysis) -> list[Number]:
    """Results, "Inter-model win rates" (Figure fig:cross_win_rate)."""
    win, n = a.net_win_rate, len(C.DATASETS)
    chem = win.loc["CheMeleon"].drop("CheMeleon")
    worst = a.order[-5:]
    return [
        Number("Datasets n", "25", str(n)),
        Number("CheMeleon has a positive net win rate against every other model", "yes", "yes" if (chem > 0).all() else "no"),
        Number("CheMeleon's smallest margin", "R-MAT, 3 more datasets won than lost",
               f"{chem.idxmin()}, {round(chem.min() * n)} more datasets won than lost"),
        Number("Among the worst-performing models (last five by mean rank)", "SELFormer, SimSon, GEM and ChemGPT",
               and_join(worst), {"SELFormer", "SimSon", "GEM", "ChemGPT"} <= set(worst)),
    ]


def dataset_winrate(a: Analysis) -> list[Number]:
    """Results, "Per-dataset wins against ECFP" (Figure fig:dataset_winrate)."""
    n_beat = a.dataset_winrate.set_index("dataset")["sum"]
    return [Number("Datasets on which no model exceeds ECFP by more than 0.01", "8 of the 25",
                   f"{(n_beat == 0).sum()} of the {len(n_beat)}")]


def near_wins(a: Analysis) -> list[Number]:
    """Results, Table tab:close_to_winning."""
    nw = a.near_wins
    second = nw[nw["n_datasets"] == nw["n_datasets"].iloc[1]]
    return [
        Number("Most wins or near-wins", "CheMeleon, 12 datasets", f"{nw.iloc[0]['family']}, {nw.iloc[0]['n_datasets']} datasets"),
        Number("Next", "8 for ECFP, MolBERT, Mordred and R-MAT",
               f"{second['n_datasets'].iloc[0]} for {and_join(second['family'])}"),
    ]


def posthoc_raw(a: Analysis) -> list[Number]:
    """Results, "External CheMeleon validation" (Table tab:posthoc_raw)."""
    r = a.posthoc_raw
    gap = (r["CheMeleon_mean"] - r["ECFP_mean"]).abs()
    above_min = r.loc[gap >= r[["CheMeleon_std", "ECFP_std"]].min(axis=1), "dataset_name"]
    above_mean = r.loc[gap >= r[["CheMeleon_std", "ECFP_std"]].mean(axis=1), "dataset_name"]
    return [
        Number("Datasets where the difference of the means is not smaller than both standard deviations over folds",
               "none", and_join(above_min)),
        Number("... than the mean of the two standard deviations", "none", and_join(above_mean)),
    ]


def posthoc_shrinkage(a: Analysis) -> list[Number]:
    """Results, Figure fig:posthoc_forest: per-dataset posteriors at rho = 0 (and 0.2)."""
    p0, p2 = (a.per_dataset(r).set_index("dataset_name") for r in ("0", "0.2"))
    outside = p0.loc[p0["shrunk_mean"].abs() > C.TIE, "shrunk_mean"].sort_values(ascending=False).index
    return [
        Number("rho = 0: every 89% HDI includes zero", "yes", "yes" if ((p0.hdi_low < 0) & (p0.hdi_high > 0)).all() else "no"),
        Number("Eye corrosion, raw difference shrunk to (rho = 0)", "+2.93% -> +1.20%",
               f"{signed_pct(p0.loc['Eye corrosion', 'raw_mean'])} -> {signed_pct(p0.loc['Eye corrosion', 'shrunk_mean'])}"),
        Number("Mitochondrial toxicity, raw difference shrunk to (rho = 0)", "+3.22 -> +1.05",
               f"{100 * p0.loc['Mitochondrial toxicity', 'raw_mean']:+.2f} -> "
               f"{100 * p0.loc['Mitochondrial toxicity', 'shrunk_mean']:+.2f}"),
        Number("rho = 0: posterior means outside the ROPE", "Eye corrosion and Mitochondrial toxicity", and_join(outside)),
        Number("rho = 0.2: Mitochondrial toxicity shrunk to", "+0.84", f"{100 * p2.loc['Mitochondrial toxicity', 'shrunk_mean']:+.2f}"),
        Number("rho = 0.2: every posterior mean within the ROPE", "yes", "yes" if (p2["shrunk_mean"].abs() <= C.TIE).all() else "no"),
    ]


def posthoc_rho_sweep(a: Analysis) -> list[Number]:
    """Results, Table tab:posthoc_rho and Figure fig:posthoc_simplex: rho from 0 to 0.2."""
    s = a.posthoc_stats
    rng = lambda c, d=1: f"{100 * s[c].min():.{d}f}% to {100 * s[c].max():.{d}f}%"  # noqa: E731
    return [
        Number("Decision for every rho", "inconclusive", "inconclusive" if (s["decision"] == "Unknown").all()
               else ", ".join(s["decision"])),
        Number("P(equivalent)", "66.2% to 68.1%", rng("p_rope")),
        Number("P(CheMeleon better)", "31.1% to 33.3%", rng("p_right")),
        Number("E[mu_0]", "+0.65% to +0.74%", f"{signed_pct(s['estimate'].min())} to {signed_pct(s['estimate'].max())}"),
        Number("Every 89% HDI of mu_0 includes zero", "yes", "yes" if ((s.hdi_low < 0) & (s.hdi_high > 0)).all() else "no"),
        Number("CheMeleon at least as good as ECFP on a new dataset, 1 - P(ECFP better), for every rho", "at least 98.7%",
               f"{1 - s['p_left'].max():.2%}", 1 - s["p_left"].max() >= 0.987),
    ]


def posthoc_vs_bbt(a: Analysis) -> list[Number]:
    """Results, "External CheMeleon validation": the tie rule of BBT vs the t-test."""
    r = a.posthoc_raw
    d = r["CheMeleon_mean"] - r["ECFP_mean"]
    wins, ties, losses = a.head_to_head("CheMeleon")
    steady = r.nsmallest(4, "diff_std")
    return [
        Number("CheMeleon vs ECFP on the 25 benchmark datasets", "12 wins, 11 ties, 2 losses", f"{wins} wins, {ties} ties, {losses} losses"),
        Number("CheMeleon vs ECFP on the external datasets (mean AUROC over folds)", "7 wins, 4 ties, 0 losses",
               f"{(d > C.TIE).sum()} wins, {(d.abs() <= C.TIE).sum()} ties, {(d < -C.TIE).sum()} losses"),
        Number("Datasets where the std of the per-fold differences does not exceed their mean", "Eye corrosion",
               and_join(r.loc[r["diff_std"] <= r["diff_mean"].abs(), "dataset_name"])),
        Number("Four datasets with the least varying differences (by std)", "B3DB, DILIrank 2.0, bioavailability and NICEATM",
               and_join(map(short_name, steady["dataset_name"])),
               {short_name(n).lower() for n in steady["dataset_name"]} == {"b3db", "dilirank 2.0", "bioavailability", "niceatm"}),
        Number("... their mean difference", "+0.16%", signed_pct(steady["diff_mean"].mean())),
    ]


def compute_cost(a: Analysis) -> list[Number]:
    """Results, "Computational cost" (Figure fig:compute_times, Table tab:runtime_top)."""
    rt = a.runtime.set_index("family")
    fp = rt.loc[["ECFP", "Atom Pair", "TT"], "mean_runtime"]
    nn = rt.drop(index=C.FINGERPRINTS)["mean_runtime"].sort_values()
    chem = rt.loc["CheMeleon"]
    top_nn = [f for f in a.order if f not in C.FINGERPRINTS][:4]
    ratios = [rt.loc[m, "mean_runtime"] / chem["mean_runtime"] for m in ["R-MAT", "MolBERT"]]
    return [
        Number("Fastest neural network, seconds per 10,000 molecules", "MolE, 6.37", f"{nn.index[0]}, {nn.iloc[0]:.2f}"),
        Number("Slowest fingerprint (ECFP, AP, TT), seconds", "less than one second", f"{fp.max():.2f}", fp.max() < 1),
        Number("CheMeleon, seconds", "9.88 ± 0.51", f"{chem['mean_runtime']:.2f} ± {chem['std_runtime']:.2f}"),
        Number("Neural networks faster than CheMeleon", "MolE", and_join(nn[nn < chem["mean_runtime"]].index)),
        Number("CheMeleon is the fastest of the top-performing models", "yes",
               "yes" if rt.loc[top_nn, "mean_runtime"].idxmin() == "CheMeleon" else "no"),
        Number("CheMeleon faster than R-MAT and MolBERT by a factor of", "about 25",
               " and ".join(f"{x:.1f}" for x in ratios), all(round(x / 5) * 5 == 25 for x in ratios)),
    ]


# --------------------------------------------------------------------------- supplement


def dataset_curation(a: Analysis) -> list[Number]:
    """Supplement "Datasets": molecules removed from MoleculeNet_HIV."""
    d = load_benchmark_datasets(a.results).groupby("dataset").first()
    hiv = d.loc["ogbg-molhiv"]
    selfies = (hiv["n_removed"] - hiv["n_removed_dative"]) / hiv["n_molecules_raw"]
    dative = d["n_removed_dative"].sum() / hiv["n_molecules_raw"]
    other = d.drop(index="ogbg-molhiv")["n_removed"].sum()
    return [
        Number("HIV molecules removed for SELFIES parsing errors", "less than 0.01%",
               f"{selfies:.3%} ({hiv['n_removed'] - hiv['n_removed_dative']} of {hiv['n_molecules_raw']})", selfies < 1e-4),
        Number("Molecules with dative bonds removed (share of HIV)", "less than 0.003%",
               f"{dative:.4%} ({d['n_removed_dative'].sum()} molecule)", dative < 3e-5),
        Number("Molecules removed from the other 24 datasets", "none", "none" if other == 0 else str(other)),
    ]


def model_variants(a: Analysis) -> list[Number]:
    """Supplement "Model variants": number of variants per model."""
    article = {"Atom Pair": 2, "CDDD": 1, "ChemBERTa": 6, "CheMeleon": 1, "ChemFM": 2, "Chemformer": 3, "ChemGPT": 2,
               "CLAMP": 1, "COATI": 1, "ContextPred": 1, "ECFP": 2, "GEM": 1, "GraphFP": 2, "GraphMVP": 6,
               "GROVER": 2, "MAT": 3, "Mol2Vec": 1, "MolBERT": 1, "MolE": 1, "MoLFormer": 1, "MolR": 4,
               "Mordred": 1, "R-MAT": 1, "SELFormer": 2, "SimSon": 1, "TT": 2, "Uni-Mol": 1, "Uni-Mol2": 1}
    family = {e: C.REP_LABELS.get(e, C.MODELS[e][0]) for e in a.results_all["embedder"].unique()}
    counts = pd.Series(family).value_counts()
    vs = a.variant_selection[a.variant_selection["n_variants"] > 1]
    differ = vs[~vs["same"]]
    return [
        Number("Variants of the models", "53 variants of the 28 models", f"{counts.sum()} variants of the {len(counts)} models"),
        Number("Variants per model", ", ".join(f"{m} ({n})" for m, n in sorted(article.items())),
               ", ".join(f"{m} ({counts.get(m, 0)})" for m in sorted(article))),
        Number("Models with more than one variant", "13", str(len(vs))),
        Number("Models where highest mean AUROC and highest BBT strength pick the same variant", "11", str(vs["same"].sum())),
        Number("... and where they differ (BBT pick instead of mean-AUROC pick)",
               "ChemGPT ([19M] instead of [4.7M]) and TT ([Count] instead of binary)",
               and_join(f"{r.family} ({C.MODELS[r.bbt_variant][1].removeprefix(r.family).strip() or 'binary'} instead of "
                        f"{C.MODELS[r.auroc_variant][1].removeprefix(r.family).strip() or 'binary'})"
                        for r in differ.itertuples())),
    ]


def runtime_supplement(a: Analysis) -> list[Number]:
    """Supplement "Computation times" (Table sup:tab:runtime_full)."""
    rt = a.runtime.set_index("family")
    nn = rt.drop(index=C.FINGERPRINTS)["mean_runtime"]
    raw = load_runtime(a.results, keep_all=True)
    mat = raw[raw["embedder"] == rt.loc["MAT", "timed_embedder"]].set_index("n_full_batches")
    ninth = 9 * mat.loc[9, "mean_runtime"] - 8 * mat.loc[8, "mean_runtime"]  # batch 9 = sum of 9 - sum of 8
    return [
        Number("Full batches per model", "nine", {9: "nine"}.get(rt["n_full_batches"].max(), str(rt["n_full_batches"].max()))),
        Number("MAT: mean time of the 8 kept batches, time of the excluded batch 9 (s)", "about 23 to 93",
               f"about {mat.loc[8, 'mean_runtime']:.0f} to {ninth:.0f}"),
        Number("MAT: conformer generation of the one slow molecule (s)", "73", None),
        Number("MolE and CheMeleon (s)", "6.37 and 9.88",
               f"{rt.loc['MolE', 'mean_runtime']:.2f} and {rt.loc['CheMeleon', 'mean_runtime']:.2f}"),
        Number("Slowest model (s)", "2041 (Uni-Mol2)", f"{nn.max():.0f} ({nn.idxmax()})"),
        Number("Neural networks span more than two orders of magnitude (slowest / fastest)", "more than 100",
               f"{nn.max() / nn.min():.0f}", nn.max() / nn.min() > 100),
    ]


def bbt_strong(a: Analysis) -> list[Number]:
    """Supplement "BBT strong interpretation" (Figure sup:fig:bbt_strong_interpreatation, Table sup:tab:bbt_strong_decisions)."""
    bt = a.bbt_vs_ecfp.set_index("other")
    s = bt["decision_strong"]
    worse = s[s == "Worse"].index.tolist()
    no_claim = s[s.str.startswith("Unknown")].index.tolist()
    claimed = s[~s.str.startswith("Unknown")]
    same = (claimed == bt.loc[claimed.index, "decision"]).sum()
    chem = bt.loc["CheMeleon"]
    return [
        Number("Better than ECFP", "none", and_join(s[s == "Better"].index)),
        Number("Practically equivalent", "MolBERT, ChemBERTa, Mordred, CDDD, MAT, Atom Pair and MolE",
               and_join(s[s == "Practically equivalent"].index)),
        Number("Worse: number, from", "14, from ChemFM", f"{len(worse)}, from {worse[0]}"),
        Number("ChemFM, posterior mean pi", "0.295", f"{bt.loc['ChemFM', 'p_other']:.3f}"),
        Number("No claim", "CheMeleon, R-MAT, MoLFormer, Mol2Vec, CLAMP and TT", and_join(no_claim)),
        Number("CheMeleon, posterior mean pi", "0.64", f"{chem['p_other']:.2f}"),
        Number("CheMeleon, 89% HDI of pi lies within (0.55, 0.70)", "yes",
               f"[{chem['p_other_hdi_low']:.3f}, {chem['p_other_hdi_high']:.3f}]",
               0.55 < chem["p_other_hdi_low"] and chem["p_other_hdi_high"] < 0.70),
        Number("Models with a strong-interpretation decision, all equal to the weak one", "21",
               str(len(claimed)), len(claimed) == 21 and same == len(claimed)),
        Number("Models without a claim under the strong interpretation", "six", {6: "six"}.get(len(no_claim), str(len(no_claim)))),
    ]


def nu_prior_sensitivity(a: Analysis) -> list[Number]:
    """Supplement "Sensitivity of the hierarchical t-test to the prior on nu" (Table sup:tab:nu_prior, Figure sup:fig:nu_prior)."""
    from scipy import stats as st

    s = a.ttest_stats.set_index(["rho_label", "prior"])
    main, js = s.xs("corani", level="prior"), s.xs("juarez_steel", level="prior")
    prior = st.gamma(C.JS_SHAPE, scale=1 / C.JS_RATE)
    post = js.loc[C.POSTHOC_MAIN_RHO]
    max_p = max((main[c] - js[c]).abs().max() for c in ["p_left", "p_rope", "p_right"])
    per_ds = max((a.per_dataset(r)["shrunk_mean"] - a.per_dataset(r, "juarez_steel")["shrunk_mean"]).abs().max()
                 for r in C.POSTHOC_RHOS)
    return [
        Number("External datasets", "11", str(len(C.POSTHOC_DATASETS))),
        Number("Juárez & Steel prior: median [5%, 95%]", "16.8 [3.6, 47.4]",
               f"{prior.median():.1f} [{prior.ppf(0.05):.1f}, {prior.ppf(0.95):.1f}]"),
        Number("Juárez & Steel posterior at rho = 0.2: median [5%, 95%]", "17.9 [4.5, 48.4]",
               f"{post.nu_median:.1f} [{post.nu_q05:.1f}, {post.nu_q95:.1f}]"),
        Number("Max |E[mu_0] difference| between the priors", "0.0001", f"{(main['estimate'] - js['estimate']).abs().max():.4f}"),
        Number("Max |probability difference| between the priors", "0.010", f"{max_p:.3f}"),
        Number("... in Monte Carlo standard errors of P(equivalent) (median over the 12 fits)", "about 1.6",
               f"{max_p / a.ttest_stats['mcse_p_rope'].median():.2f}",
               abs(max_p / a.ttest_stats["mcse_p_rope"].median() - 1.6) < 0.1),
        Number("Max |per-dataset posterior mean difference|", "0.0002", f"{per_ds:.4f}"),
        Number("Decision under both priors, every rho", "inconclusive",
               "inconclusive" if (a.ttest_stats["decision"] == "Unknown").all() else "differs"),
        Number("All 12 fits: max R-hat", "at most 1.01", f"{a.ttest_stats['max_r_hat'].max():.2f}",
               a.ttest_stats["max_r_hat"].max() <= 1.01),
        Number("All 12 fits: min bulk ESS", "at least 5,500", f"{a.ttest_stats['min_ess_bulk'].min():,.0f}",
               a.ttest_stats["min_ess_bulk"].min() >= 5500),
        Number("All 12 fits: max divergent transitions (of 20,000 draws)", "4", f"{a.ttest_stats['divergences'].max():.0f}"),
    ]


MAIN = [scope, variant_selection, rope_beta_bound, mean_rank, chemberta_mtr, bbt_decisions, bbt_convergence, cdd,
        net_win_rate, dataset_winrate, near_wins, posthoc_raw, posthoc_shrinkage, posthoc_rho_sweep, posthoc_vs_bbt,
        compute_cost]
SUPPLEMENT = [dataset_curation, model_variants, runtime_supplement, bbt_strong, nu_prior_sensitivity]


def to_markdown(results: dict[str, list[tuple[str, list[Number]]]]) -> str:
    """numbers.md: one table per function, a mark on every number that differs from the article."""
    flat = [n for groups in results.values() for _, ns in groups for n in ns]
    bad = [n for n in flat if n.ok is False]
    lines = ["# In-text numbers of the article", "",
             "Computed from the results directory by `make_figures.py`. ✗ = differs from the article "
             "(main.tex / supplementary.tex), – = not derivable from the results.", "",
             f"{len(flat)} numbers: {sum(n.ok is True for n in flat)} agree, {len(bad)} differ, "
             f"{sum(n.ok is None for n in flat)} not derivable.", ""]
    if bad:
        lines += ["## Differences", "", "| Statement | Article | Computed |", "|---|---|---|",
                  *[f"| {n.statement} | {n.article} | {n.computed} |" for n in bad], ""]
    for part, groups in results.items():
        lines += [f"## {part}", ""]
        for doc, ns in groups:
            lines += [f"### {doc}", "", "| Statement | Article | Computed | |", "|---|---|---|---|"]
            lines += [f"| {n.statement} | {n.article} | {'–' if n.computed is None else n.computed} | "
                      f"{'✓' if n.ok else '–' if n.ok is None else '✗'} |" for n in ns]
            lines.append("")
    return "\n".join(lines)
