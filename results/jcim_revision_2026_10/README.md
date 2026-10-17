# Results of the JCIM revision (2026-10)

The only input of `published_figures/make_figures.py`: the outcomes of the experiments of the article.

| File | Content |
|---|---|
| `all.db` | The benchmark: table `classificationreport` holds the test AUROC of every embedding model x dataset x classifier (knn, rf, ridge) on the 25 benchmark datasets. The article's CLAMP is `CLAMP_refitted`, the leak-free retraining; `CLAMP` holds the authors' original weights, which saw benchmark molecules during pretraining, and is not used. The other 53 - 1 variants are those of `src/config.py:MODELS`. |
| `posthoc_scaffold.csv` | The external validation: AUROC on every fold of the 5-fold scaffold cross-validation of the 11 external datasets (SMILES-deduplicated), per embedder and classifier; the article uses `chemeleon` and `ecfp_count` (kNN on raw counts with Tanimoto). |
| `runtime.csv` | Embedding time per 10,000 MUV molecules. Append-only: the last row of an embedder wins (MAT: 8 full batches, batch 9 dropped). CLAMP was timed with the authors' checkpoint (row `CLAMP`), which has the same architecture as the retrained model. |
| `external_datasets/` | The 11 external datasets after curation (`smiles`, `label`, ...) and notes on their sources and preprocessing (`data_sources.txt`). |
| `benchmark_datasets.csv` | Per benchmark dataset and task: molecules before and after removing unparsable molecules, removed molecules with dative bonds, positive / negative labels. |
