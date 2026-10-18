# Figures, tables and numbers of the revision

Reproduces every figure, table and in-text number of the revision of *Benchmarking Pretrained Molecular
Embedding Models for Molecular Representation Learning* (arXiv:2508.06199): main text and Supporting Information.
Only objects of the final revision are produced; the TOC graphic (visual abstract) is not generated.

## Run

From the repo root (~10 min; 4 cores for the BBT fits, `--jobs 12` x `--cores 4` = 48 cores for the 12 t-test fits):

```bash
~/.local/bin/uv run --no-sync --with-editable published_figures/bayes-testkit python published_figures/make_figures.py
```

- `--results DIR`: results directory, default `results/revision_2026_10/` (the only input, see below).
- `--out DIR`: output directory, default `published_figures/output/`.
- `--only tables figures numbers`: generate some object types only.
- `--check-tex main.tex supplementary.tex`: compare the generated tables with the manuscripts' (whitespace ignored).
- The hierarchical t-test fits are pickled in `<out>/cache/` and reused while the fold scores and sampler settings
  are unchanged (~1 min instead of ~7); `--refit` forces new fits. The BBT fits (~3 min) are always rerun.

## Input

The results directory holds the outcomes of the experiments described in the article:

- `all.db`: the benchmark, test AUROC of every model variant on every dataset with each classifier (kNN, RF, LR).
  CLAMP is `CLAMP_refitted`, the leak-free retraining; the authors' original `CLAMP` is not used.
- `posthoc_scaffold.csv`: the external validation, AUROC of CheMeleon and ECFP on every fold of the 5-fold scaffold
  cross-validation of the 11 external datasets, with each classifier.
- `runtime.csv`: embedding times of every model on MUV, in batches of 10,000 molecules (CLAMP: the authors'
  checkpoint, same architecture as the retrained model).
- `external_datasets/`: the 11 external datasets after curation, with notes on their sources and preprocessing.
- `benchmark_datasets.csv`: label counts of the 25 benchmark datasets per task, after removing unparsable molecules.

## Output (paths as `main.tex` / `supplementary.tex` include them)

| Article object | File | Function |
|---|---|---|
| Fig. `workflow_fig` | `figures/workflow.png` | `figures.workflow` (static drawing, `static/workflow.png`) |
| Fig. `fig:bbt_cdd` | `figures/updated/bbt_cdd.png` | `figures.bbt_cdd` |
| Fig. `fig:cross_win_rate` | `figures/updated/cross_winrate.png` | `figures.cross_winrate` |
| Fig. `fig:dataset_winrate` | `figures/updated/dataset_winrate.png` | `figures.dataset_winrate` |
| Fig. `fig:posthoc_forest` | `figures/updated/scaffold_rho_0_forest.png` | `figures.posthoc_forest` |
| Fig. `fig:posthoc_simplex` | `figures/updated/scaffold_rho_{0,0.2}_simplex.png` | `figures.posthoc_simplex` |
| Fig. `fig:compute_times` | `figures/updated/muv_compute_times.png` | `figures.compute_times` |
| Fig. `sup:fig:bbt_strong_interpreatation` | `figures/updated/bbt_posterior_strong.png` | `figures.bbt_posterior_strong` |
| Fig. `sup:fig:nu_prior` | `figures/updated/juarez_steel_prior.png` | `figures.nu_prior` |
| Tab. `tab:external_datasets` | `tables/tab_external_datasets.tex` | `tables.external_datasets` (descriptive) |
| Tab. `tab:mean_rank` | `tables/tab_mean_rank.tex` | `tables.mean_rank` |
| Tab. `tab:bbt_decisions` | `tables/tab_bbt_decisions.tex` | `tables.bbt_decisions` |
| Tab. `tab:close_to_winning` | `tables/tab_close_to_winning.tex` | `tables.close_to_winning` |
| Tab. `tab:posthoc_raw` | `tables/tab_posthoc_raw.tex` | `tables.posthoc_raw` |
| Tab. `tab:posthoc_rho` | `tables/tab_posthoc_rho.tex` | `tables.posthoc_rho` |
| Tab. `tab:runtime_top` | `tables/tab_runtime_top.tex` | `tables.runtime_top` |
| Tab. `sup:tab:data_info` | `tables/sup_tab_data_info.tex` | `tables.data_info` |
| Tab. `sup:tab:external_data_info` | `tables/sup_tab_external_data_info.tex` | `tables.external_data_info` |
| Tab. `sup:tab:variant_selection` | `tables/sup_tab_variant_selection.tex` | `tables.variant_selection` |
| Tab. `sup:tab:runtime_full` | `tables/sup_tab_runtime_full.tex` | `tables.runtime_full` |
| Tab. `sup:tab:bbt_rope_values` | `tables/sup_tab_bbt_rope_values.tex` | `tables.bbt_rope_values` |
| Tab. `sup:tab:bbt_strong_decisions` | `tables/sup_tab_bbt_strong_decisions.tex` | `tables.bbt_strong_decisions` |
| Tab. `sup:tab:bbt_full` | `tables/sup_tab_bbt_full.tex` | `tables.bbt_full` |
| Tab. `sup:tab:nu_prior` | `tables/sup_tab_nu_prior.tex` | `tables.nu_prior` |
| Tab. `sup:tab:full_comparison` | `tables/sup_tab_full_comparison.tex` | `tables.full_comparison` |
| Tab. `sup:tab:close_to_winning` | `tables/sup_tab_close_to_winning.tex` | `tables.sup_close_to_winning` |
| In-text numbers | `numbers.md`, `numbers.csv` | `numbers.*` (one function per group) |

Every table also has a `.csv` with its values (`fig_*.csv`: values behind three of the figures). `numbers.md` lists
each number as printed in the article next to the recomputed value and marks every disagreement.

## Code

- `make_figures.py`: entry point; calls every function of `src/tables.py`, `src/figures.py`, `src/numbers.py`.
- `src/config.py`: datasets, the 53 variants and their names, protocol and test settings.
- `src/data.py`: readers of the results directory. `src/analysis.py`: shared computations (best head,
  representatives, BBT fits, t-test fits and their cache), computed once on first use.
- `src/latex.py`: table layout helpers.

## Protocol (as in the article)

- Best classification head (max test AUROC over kNN, RF, LR) per dataset and variant; AUROC differences of at most
  0.01 are ties (BBT add rule), near-wins and "outperforms ECFP" use the same threshold.
- BBT (`BBTTest(absolute_tie_threshold=0.01, tie_solver="add")`, σ ~ LogNormal(0, 0.5), Σβ = 0, seed 42): a fit on
  all 53 variants (40000 draws, 10000 tune) selects each model's variant with the highest posterior mean β
  (fingerprints: count variant, fixed by design); a refit on the 28 representatives (10000 / 2000,
  `target_accept=0.95`) gives every comparison. Weak interpretation with ROPE (0.35, 0.65) in the main text, strong
  interpretation in the supplement.
- External validation: CheMeleon vs ECFP count, best head by mean AUROC over the 5 scaffold folds,
  `HierarchicalTTest(rope=0.01, parametrization="non_centered")`, 4 x 5000 draws (2000 tune), seed 42, for
  ρ ∈ {0 (fitted as 1e-9), 0.001, 0.01, 0.05, 0.1, 0.2 = 1/k}; supplement: the same with ν ~ Gamma(2, 0.1).
- Naming follows the manuscripts: the main text writes "Mol2vec" (`config.MAIN_TEXT_NAMES`), the supplement and
  the figures "Mol2Vec"; the main-text t-test table says "Inconclusive" where the supplement says "Unknown".
