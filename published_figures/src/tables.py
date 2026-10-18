"""One function per table of the article, written to <out>/tables/<label>.tex (+ .csv with the values).

The LaTeX follows the manuscripts' own layout, float placement, caption and label, so a table can replace its
counterpart in main.tex / supplementary.tex verbatim. File names are the labels with ':' -> '_'.
"""

from pathlib import Path

import pandas as pd

from . import config as C
from .analysis import Analysis, bbt_vs_baseline
from .data import load_benchmark_datasets, load_external_labels
from .latex import boxed_table, pct, tex_escape, tex_table, write_tex


def main_name(model: str) -> str:
    """A model's name in the main text (Mol2vec); the supplement uses the names of config.MODELS."""
    return C.MAIN_TEXT_NAMES.get(model, model)


def decisions_tex(groups: dict[str, list[str]], caption: str, label: str, cut_worse: bool, per_line: int = 3) -> str:
    """Decision -> models table; `cut_worse` shortens "Worse" to its first three models, as the main text does."""
    def wrap(names):
        return ", \\\\ ".join(", ".join(names[i:i + per_line]) for i in range(0, len(names), per_line))

    def short(d, m):
        return ", ".join(m[:3]) + ", ..., \\\\ (all other models)" if cut_worse and d == "Worse" and len(m) > 3 else wrap(m)

    def cell(d):  # two-line labels: "Practically / equivalent", "Unknown / (worse/equivalent)"
        two = d.replace("Practically ", "Practically\\\\ ").replace("Unknown (", "Unknown\\\\ (")
        return d if d == two else f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{two}\\end{{tabular}}"

    rows = [f"{cell(d)} & \\begin{{tabular}}[c]{{@{{}}c@{{}}}} {tex_escape(short(d, m)) if m else 'None'} \\end{{tabular}} "
            f"\\\\ \\hline" for d, m in groups.items() if m or d != "Unknown"]
    rows[-1] = rows[-1].removesuffix(" \\hline")
    return tex_table("table", "c|c", "\\textbf{BBT decision} & \\textbf{Models} \\\\", rows, caption, label,
                     placement="[H]")


def write(out: Path, label: str, tex: str, csv: pd.DataFrame) -> None:
    name = label.replace(":", "_")
    csv.to_csv(out / C.TABLES / f"{name}.csv", index=False)
    write_tex(out / C.TABLES / f"{name}.tex", tex)


# --------------------------------------------------------------------------- main text


def external_datasets(a: Analysis, out: Path) -> None:
    """Table tab:external_datasets: the 11 external validation datasets, references and endpoints (descriptive)."""
    t = pd.DataFrame({"dataset": list(C.POSTHOC_DATASETS), "name": list(C.POSTHOC_DATASETS.values())})
    t["cite"] = t["dataset"].map(lambda d: C.POSTHOC_DATASET_CITES[d].replace(",", ", "))
    t["endpoint"] = t["dataset"].map(C.POSTHOC_DATASET_ENDPOINTS)
    write(out, "tab:external_datasets", boxed_table(
        "|l|l|p{0.48\\linewidth}|", "\\textbf{Dataset} & \\textbf{Reference} & \\textbf{Endpoint} \\\\",
        [f"{r.name} & \\cite{{{r.cite}}} & {r.endpoint} \\\\" for r in t.itertuples()],
        "Selected external datasets used for evaluation.", "tab:external_datasets"), t)


def mean_rank(a: Analysis, out: Path) -> None:
    """Table tab:mean_rank: mean rank and mean AUROC of the 28 models, ECFP row in bold between rules."""
    t = a.mean_rank
    rows = []
    for r in t.itertuples():
        cells = [tex_escape(main_name(r.family)), f"{r.mean_rank:.2f}", tex_escape(pct(r.mean_auroc))]
        if r.family == "ECFP":
            rows[-1] += " \\hline"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        rows.append(" & ".join(cells) + (" \\\\ \\hline" if r.family == "ECFP" else " \\\\"))
    write(out, "tab:mean_rank", tex_table(
        "table", "l|c|c", "\\textbf{Model} & \\textbf{Mean rank} $\\downarrow$ & \\textbf{Mean AUROC} $\\uparrow$ \\\\",
        rows, "Average model results, with ECFP marked.", "tab:mean_rank", placement="[H]"), t)


def bbt_decisions(a: Analysis, out: Path) -> None:
    """Table tab:bbt_decisions: BBT decision of every model vs ECFP, ROPE (0.35, 0.65); "Worse" cut to three."""
    bt = a.bbt_vs_ecfp
    groups = {d: [main_name(m) for m in bt.loc[bt["decision"] == d, "other"]] for d in C.BBT_DECISIONS}
    write(out, "tab:bbt_decisions", decisions_tex(
        groups, "Bayesian Bradley-Terry (BBT) model decisions for all models compared to the ECFP baseline, with "
        f"ROPE $[{C.BBT_ROPE[0]}, {C.BBT_ROPE[1]}]$.", "tab:bbt_decisions", cut_worse=True),
        pd.DataFrame([{"decision": d, "models": ", ".join(m)} for d, m in groups.items()]))


def close_to_winning(a: Analysis, out: Path) -> None:
    """Table tab:close_to_winning: the five models with the most wins or near-wins (within 0.01 AUROC)."""
    nw = a.near_wins.head(5)
    write(out, "tab:close_to_winning", tex_table(
        "table", "l|r", "Model & Number of datasets \\\\",
        [f"{tex_escape(main_name(r.family))} & {r.n_datasets} \\\\" for r in nw.itertuples()],
        f"Number of datasets on which a model achieved the highest AUROC or was within {C.TIE} AUROC of it, "
        "for the five models with the most such datasets.", "tab:close_to_winning", placement="[H]"), nw)


def posthoc_raw(a: Analysis, out: Path) -> None:
    """Table tab:posthoc_raw: AUROC of CheMeleon and ECFP on the external datasets, mean ± std over 5 scaffold folds."""
    res = a.posthoc_raw
    left, right = C.POSTHOC_MODELS.values()
    n_folds = a.posthoc_folds[0]["fold"].nunique()
    cell = lambda m, s: f"{100 * m:.2f} $\\pm$ {100 * s:.2f}"  # noqa: E731
    means = res[[f"{left}_mean", f"{right}_mean"]].mean()
    write(out, "tab:posthoc_raw", boxed_table(
        "|l|c|c|", f"\\textbf{{Dataset}} & \\textbf{{{left} AUROC [\\%]}} & \\textbf{{{right} AUROC [\\%]}} \\\\",
        [f"{tex_escape(r.dataset_name)} & {cell(getattr(r, f'{left}_mean'), getattr(r, f'{left}_std'))} & "
         f"{cell(getattr(r, f'{right}_mean'), getattr(r, f'{right}_std'))} \\\\" for r in res.itertuples()],
        f"Results of the post-hoc comparison on external validation datasets. Mean $\\pm$ standard deviation is "
        f"reported across {n_folds}-fold {C.SPLIT} cross-validation. For each dataset and model, the classification "
        f"head with the best mean AUROC is used.", "tab:posthoc_raw",
        footer=[f"\\textbf{{Mean}} & \\textbf{{{100 * means.iloc[0]:.2f}}} & \\textbf{{{100 * means.iloc[1]:.2f}}} \\\\"]),
        res)


def posthoc_rho(a: Analysis, out: Path) -> None:
    """Table tab:posthoc_rho: hierarchical t-test, CheMeleon vs ECFP, for each fold correlation rho."""
    s = a.posthoc_stats
    left, right = C.POSTHOC_MODELS.values()
    decision = {"Unknown": "Inconclusive"}  # the main text's word for the library's "Unknown"
    write(out, "tab:posthoc_rho", tex_table(
        "table", "r|c|c|c|c|c|l",
        f"$\\rho$ & $\\mathbb{{E}}[\\mu_0]$ & 89\\% HDI & P({right}) & P(equivalent) & P({left}) & Decision \\\\",
        [f"{r.rho_label} & {r.estimate:+.4f} & [{r.hdi_low:+.4f}, {r.hdi_high:+.4f}] & {r.p_left:.3f} & "
         f"{r.p_rope:.3f} & {r.p_right:.3f} & {decision.get(r.decision, r.decision)} \\\\" for r in s.itertuples()],
        f"Results of the Bayesian hierarchical correlated t-test, {left} against {right}, for different values of "
        "the fold correlation $\\rho$.", "tab:posthoc_rho"),
        s[["rho_label", "estimate", "hdi_low", "hdi_high", "p_left", "p_rope", "p_right", "decision"]])


def runtime_top(a: Analysis, out: Path) -> None:
    """Table tab:runtime_top: embedding time of the five best models by mean rank."""
    tab = a.runtime.sort_values("mean_rank").head(5)
    write(out, "tab:runtime_top", "\n".join([
        "\\begin{table}[t]", "    \\centering", "\\begin{tabular}{c|c|c|c}",
        "\\textbf{Model} & \\textbf{Device} & \\textbf{Mean time [s]} $\\downarrow$ & \\textbf{Std time [s]} \\\\",
        "\\hline\\hline",
        *[f"{r.family} & {r.device.replace('cuda', 'gpu').upper()} & {r.mean_runtime:.2f} & {r.std_runtime:.2f} \\\\"
          for r in tab.itertuples()],
        "\\end{tabular}",
        "\\caption{Embedding time of the 5 best models by mean rank (Table~\\ref{tab:mean_rank}), using each model's "
        "representative variant. MUV is embedded in batches of 10{,}000 molecules.}",
        "\\label{tab:runtime_top}", "\\end{table}"]),
        tab[["family", "embedder", "timed_embedder", "mean_rank", "device", "mean_runtime", "std_runtime",
             "n_full_batches"]])


# --------------------------------------------------------------------------- supplement


def data_info(a: Analysis, out: Path) -> None:
    """Table sup:tab:data_info: the 25 benchmark datasets, number of molecules and tasks, mean share of positives."""
    d = load_benchmark_datasets(a.results)
    t = d.groupby("dataset").agg(n_samples=("n_molecules", "first"), n_tasks=("task", "size"),
                                 pos=("n_positive", "sum"), neg=("n_negative", "sum")).reset_index()
    t["positive_share"] = t["pos"] / (t["pos"] + t["neg"])
    t["benchmark"] = t["dataset"].map(lambda n: "MoleculeNet" if n.startswith("ogbg-") else "TDC")
    t["dataset"] = t["dataset"].replace(C.DATASET_NAMES)
    t = t.sort_values(["benchmark", "dataset"], key=lambda c: c.eq("MoleculeNet") if c.name == "benchmark" else c)
    t = t[["dataset", "benchmark", "n_samples", "n_tasks", "positive_share"]].reset_index(drop=True)
    write(out, "sup:tab:data_info", tex_table(
        "table*", "l|l|c|c|c",
        "Dataset & Benchmark & No. samples & No. tasks & \\begin{tabular}[c]{@{}c@{}}Mean share \\\\ of positive "
        "samples \\end{tabular} \\\\",
        # MoleculeNet datasets without the prefix (the Benchmark column has it)
        [f"{tex_escape(r.dataset.removeprefix('MoleculeNet_'))} & {r.benchmark} & {r.n_samples} & {r.n_tasks} & "
         f"{tex_escape(pct(r.positive_share))} \\\\" for r in t.itertuples()],
        "List of datasets used in the evaluation along with the general statistic in form of the number of "
        "samples, number of tasks and mean share of positive samples.", "sup:tab:data_info", placement="[h!]"), t)


def external_data_info(a: Analysis, out: Path) -> None:
    """Table sup:tab:external_data_info: size and class balance of the 11 external datasets."""
    rows, tex_rows = [], []
    for ds, name in C.POSTHOC_DATASETS.items():
        labels = load_external_labels(a.results, ds)
        share = labels.value_counts(normalize=True).sort_index()
        # binary: share of the positive class; multiclass: share of each class, in label order
        balance = pct(share.loc[1]) if len(share) == 2 else " / ".join(pct(s) for s in share)
        rows.append({"dataset": ds, "name": name, "n_samples": len(labels), "n_tasks": 1, "n_classes": len(share),
                     "class_balance": balance})
        tex_name = C.POSTHOC_DATASET_TEX_NAMES.get(ds)
        tex_name = f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}}{tex_name}\\end{{tabular}}" if tex_name else tex_escape(name)
        tex_rows.append(f"{tex_name} & \\cite{{{C.POSTHOC_DATASET_CITES[ds]}}} & {len(labels)} & 1 & {len(share)} & "
                        f"{tex_escape(balance)} \\\\")
    write(out, "sup:tab:external_data_info", tex_table(
        "table*", "l|l|c|c|c|c",
        "Dataset & Reference & No. samples & No. tasks & No. classes & \\begin{tabular}[c]{@{}c@{}}Positive class "
        "percentage \\\\ / class proportions \\end{tabular} \\\\",
        tex_rows,
        "List of external datasets used in the evaluation and key statistics. For binary classification, we "
        "report positive class percentage, and for multiclass, the percentage of each class.",
        "sup:tab:external_data_info", placement="[h!]"), pd.DataFrame(rows))


def variant_selection(a: Analysis, out: Path) -> None:
    """Table sup:tab:variant_selection: representative by highest mean AUROC vs by highest BBT strength, for the
    13 models with more than one variant."""
    multi = a.variant_selection[a.variant_selection["n_variants"] > 1]
    name = lambda e: tex_escape(C.MODELS[e][1])  # noqa: E731
    dagger = lambda f: "$^\\dagger$" if f in C.FIXED_REPRESENTATIVES else ""  # noqa: E731
    write(out, "sup:tab:variant_selection", tex_table(
        "table", "l|c|l|l", "Model & Variants & Highest mean AUROC & Highest BBT strength \\\\",
        [f"{tex_escape(r.family)}{dagger(r.family)} & {r.n_variants} & {name(r.auroc_variant)} & "
         f"{name(r.bbt_variant)} \\\\" for r in multi.itertuples()],
        "Family representative selected by the highest mean AUROC (best head per dataset) and by the highest "
        "strength $\\beta$ of a Bayesian Bradley-Terry model fitted on all variants, for the model families "
        "with more than one variant. $^\\dagger$ Variant fixed by design: the count variant for the fingerprints.",
        "sup:tab:variant_selection", placement="[H]"), multi)


def runtime_full(a: Analysis, out: Path) -> None:
    """Table sup:tab:runtime_full: embedding time of all 28 models (representative variant), fastest first."""
    tab = a.runtime.sort_values("mean_runtime").reset_index(drop=True)
    tab["variant"] = tab["embedder"].map(lambda e: C.MODELS[e][1])
    n = tab["n_full_batches"].max()
    mark = lambda r: "$^\\dagger$" if r.n_full_batches < n else ""  # noqa: E731
    write(out, "sup:tab:runtime_full", "\n".join([
        "\\begin{table}[H]", "    \\centering", "\\begin{tabular}{l|l|c|c|c}",
        "\\textbf{Model} & \\textbf{Variant} & \\textbf{Device} & \\textbf{Mean time [s]} $\\downarrow$ & "
        "\\textbf{Std time [s]} \\\\",
        "\\hline\\hline",
        *[f"{tex_escape(r.family)}{mark(r)} & {tex_escape(r.variant)} & {r.device.replace('cuda', 'gpu').upper()} & "
          f"{r.mean_runtime:.2f} & {r.std_runtime:.2f} \\\\" for r in tab.itertuples()],
        "\\end{tabular}",
        f"\\caption{{Time of embedding 10{{,}}000 molecules from the MUV dataset for all models, using each model's "
        f"representative variant. Mean and standard deviation over the {n} full batches of 10{{,}}000 molecules. "
        f"$^\\dagger$ {n - 1} batches, see text.}}",
        "\\label{sup:tab:runtime_full}", "\\end{table}"]),
        tab[["family", "variant", "embedder", "timed_embedder", "device", "mean_runtime", "std_runtime",
             "n_full_batches"]])


def bbt_rope_values(a: Analysis, out: Path) -> None:
    """Table sup:tab:bbt_rope_values: BBT decisions vs ECFP for ROPEs from (0.45, 0.55) to (0.2, 0.8)."""
    rp = []
    for rope in C.BBT_ROPES:
        r = bbt_vs_baseline(a.bbt, rope)
        rp.append({"rope": f"({rope[0]}, {rope[1]})",
                   **{d: ", ".join(r.loc[r["decision"] == d, "other"]) for d in C.BBT_DECISIONS}})
    rp = pd.DataFrame(rp)
    cell = lambda s: tex_escape(s) if s else "None"  # noqa: E731
    rows = [f"{r['rope']} & {cell(r['Better'])} & {cell(r['Practically equivalent'])} & {cell(r['Unknown'])} & "
            f"{cell(r['Worse'])} \\\\ \\hline" for _, r in rp.iterrows()]
    rows[-1] = rows[-1].removesuffix(" \\hline")
    write(out, "sup:tab:bbt_rope_values", tex_table(
        "table", "p{2cm} || p{2cm} | p{4cm} | p{2cm} | p{4cm}",
        "ROPE & Better than ECFP & Practically equivalent & Unknown & Worse than ECFP \\\\",
        rows, "BBT model interpretations for different ROPE values.", "sup:tab:bbt_rope_values",
        placement="[H]").replace("\\hline\\hline", "\\hline \\hline"), rp)


def bbt_strong_decisions(a: Analysis, out: Path) -> None:
    """Table sup:tab:bbt_strong_decisions: BBT decisions vs ECFP under the strong interpretation."""
    bt = a.bbt_vs_ecfp
    groups = {d: bt.loc[bt["decision_strong"] == d, "other"].tolist() for d in C.STRONG_DECISIONS}
    write(out, "sup:tab:bbt_strong_decisions", decisions_tex(
        groups,
        "Bayesian Bradley-Terry (BBT) model decisions under the strong interpretation \\cite{BBT}, comparing "
        "pretrained models to the ECFP baseline. Decisions are based on the posterior mean win probability "
        "$\\hat{\\pi}$: practically equivalent if $0.45 \\le \\hat{\\pi} \\le 0.55$, better if $\\hat{\\pi} > 0.70$, "
        "worse if $\\hat{\\pi} < 0.30$; otherwise unknown, between equivalent and better ($0.55 < \\hat{\\pi} \\le 0.70$) "
        "or between worse and equivalent ($0.30 \\le \\hat{\\pi} < 0.45$).",
        "sup:tab:bbt_strong_decisions", cut_worse=False, per_line=7),
        pd.DataFrame([{"decision": d, "models": ", ".join(m)} for d, m in groups.items()]))


def bbt_full(a: Analysis, out: Path) -> None:
    """Table sup:tab:bbt_full: every model vs ECFP: posterior mean, 89% HDI width, P(above 0.5), P(in ROPE)."""
    bt = a.bbt_vs_ecfp
    write(out, "sup:tab:bbt_full", tex_table(
        "table*", "c|c|c|c|c", "Pair & Mean & Delta & Above 50\\% & Within ROPE \\\\",
        [f"{tex_escape(r.left_model)} \\textgreater{{}} {tex_escape(r.right_model)} & {r.mean:.2f} & "
         f"{r.delta:.2f} & {r.above_50:.2f} & {r.in_rope:.2f} \\\\" for r in bt.itertuples()],
        f"Bayesian Bradley-Terry model statistics for the classification results. ROPE = "
        f"$[{C.BBT_ROPE[0]}, {C.BBT_ROPE[1]}]$", "sup:tab:bbt_full"),
        bt.drop(columns=["weak_interpretation_raw", "strong_interpretation_raw"]))


def nu_prior(a: Analysis, out: Path) -> None:
    """Table sup:tab:nu_prior: hierarchical t-test under the main prior on nu and Juárez & Steel's, every rho."""
    s = a.ttest_stats
    lines = ["\\begin{table}[H]", "    \\centering", "\\begin{tabular}{r|l|c|c|c|c|c|l}",
             "$\\rho$ & Prior on $\\nu$ & $\\mathbb{E}[\\mu_0]$ & P(ECFP) & P(equivalent) & P(CheMeleon) "
             "& $\\nu$, median [5\\%, 95\\%] & Decision \\\\", "\\hline\\hline"]
    labels = list(C.POSTHOC_RHOS)
    for i, label in enumerate(labels):
        for prior, prior_name in C.NU_PRIORS.items():
            r = s[(s["rho_label"] == label) & (s["prior"] == prior)].iloc[0]
            rho = label if prior == "corani" else ""
            lines.append(f"{rho:>5} & {prior_name} & {r.estimate:+.4f} & {r.p_left:.3f} & {r.p_rope:.3f} "
                         f"& {r.p_right:.3f} & {r.nu_median:.1f} [{r.nu_q05:.1f}, {r.nu_q95:.1f}] & {r.decision} \\\\")
        if i < len(labels) - 1:
            lines.append("\\hline")
    lines += ["\\end{tabular}",
              "\\caption{Bayesian hierarchical correlated t-test of CheMeleon against ECFP under two priors on the "
              "degrees of freedom $\\nu$.}", "\\label{sup:tab:nu_prior}", "\\end{table}"]
    write(out, "sup:tab:nu_prior", "\n".join(lines),
          s[["prior", "rho_label", "estimate", "hdi_low", "hdi_high", "p_left", "p_rope", "p_right", "decision",
             "nu_median", "nu_q05", "nu_q95", "divergences", "max_r_hat", "min_ess_bulk"]])


def full_comparison(a: Analysis, out: Path) -> None:
    """Table sup:tab:full_comparison: mean rank and mean AUROC of all 53 variants, best head and each head."""
    full = a.full_comparison
    heads = ["best", *C.HEADS.values()]
    rows = [f"{r['variant']} & " + " & ".join(f"{r[f'rank_{h}']:.2f}" for h in heads) + " & "
            + " & ".join(tex_escape(pct(r[f'auroc_{h}'])) for h in heads) + " \\\\" for _, r in full.iterrows()]
    # adjustbox shrinks the 53 rows to one page; plain `table` with [H] (the supplement is one-column)
    write(out, "sup:tab:full_comparison", tex_table(
        "table", "l|c|c|c|c||c|c|c|c",
        "Model & \\multicolumn{4}{c||}{Mean rank} & \\multicolumn{4}{c}{Mean ROC AUC score} \\\\\n"
        "Classification head & " + " & ".join(heads * 2) + " \\\\",
        rows, "Full comparison of averaged model performance including all variants and all classification heads.",
        "sup:tab:full_comparison", placement="[H]")
        .replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth, max totalheight=\\textheight}\n"
                 "\\begin{tabular}", 1)
        .replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1), full)


def sup_close_to_winning(a: Analysis, out: Path) -> None:
    """Table sup:tab:close_to_winning: wins or near-wins (within 0.01 AUROC) of all 28 models."""
    nw = a.near_wins
    write(out, "sup:tab:close_to_winning", tex_table(
        "table", "l|r", "Model & Number of datasets \\\\",
        [f"{tex_escape(r.family)} & {r.n_datasets} \\\\" for r in nw.itertuples()],
        "Number of datasets, where a given model won or had a near-win.", "sup:tab:close_to_winning",
        placement="[H]"), nw)


MAIN = [external_datasets, mean_rank, bbt_decisions, close_to_winning, posthoc_raw, posthoc_rho, runtime_top]
SUPPLEMENT = [data_info, external_data_info, variant_selection, runtime_full, bbt_rope_values, bbt_strong_decisions,
              bbt_full, nu_prior, full_comparison, sup_close_to_winning]
