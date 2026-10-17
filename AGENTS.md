# Agent Guide — Benchmarking Molecular Embedding Models

This repository benchmarks pretrained molecular embedding models. Agents should be able to **add a new model**, **run a minimal smoke test on BBBP**, and **report success/failure** without modifying core Hydra configs.

Canonical docs for humans: [`readme.md`](readme.md), [`docs/custom_model.md`](docs/custom_model.md). Prefer this file when the README is stale relative to the `uv` workflow.

## Non-negotiables

- **Env manager is `uv`** (not poetry / pyenv / `install_deps.sh`).
- **Do not edit** `config/embed.yaml` or `config/score.yaml` for local/dev testing. Use **CLI overrides** or a **throwaway / model-specific** Hydra config under `config/experiment/`.
- **Smoke-test dataset** is BBBP: Hydra config group `clf_ogbg-molbbbp` (MoleculeNet BBBP via OGB; smallest dataset in this repo).
- **Smoke-test results DB**: `data/temp_dev.db` (never write smoke-test results into `data/meta.db`).
- **Multi-variant experiments** (e.g. fingerprints via `glob(fingerprints_*)`): test **all** variants on BBBP before claiming success.

## Repository layout

```
.
├── config/                 # Hydra configs
│   ├── embed.yaml          # embedding entry config (multirun over datasets)
│   ├── score.yaml          # scoring entry config (multirun + joblib launcher)
│   ├── downloader.yaml     # dataset download entry config
│   ├── embedding/          # paths: data dirs + SQLite DB location
│   ├── dataset/            # one YAML per benchmark dataset
│   ├── experiment/         # one YAML per model family (selected via +experiment=)
│   └── model/              # per-variant model configs (swept by experiment)
├── model_wrappers/         # per-model code + isolated venvs
│   └── <model>/
│       ├── wrapper.py      # get_embedder() + SmilesEmbedder|GraphEmbedder
│       ├── init.sh         # model deps; exports HYDRA_EXPERIMENT
│       ├── .python-version # e.g. 3.11
│       └── venv/           # created by embed_wrapper.sh (gitignored)
├── src/
│   ├── common/             # Dataset/Embedder types, SQLite (peewee), utils
│   ├── embedding/          # embed pipeline
│   └── eval/               # supervised heads (rf, ridge, knn) + grid search
├── embed.py                # embedding entrypoint (do not call bare; see below)
├── score.py                # scoring entrypoint (root env)
├── download.py             # prepare datasets (root env)
├── embed_wrapper.sh        # create model venv → uv run embed.py (+ forwarded Hydra args)
├── run_embed.sh            # background embed helper
├── run_scoring.sh          # background score helper
├── pyproject.toml          # root (scoring/download) dependencies — source of truth
└── docs/custom_model.md    # how to add a model
```

Data (created at runtime, mostly gitignored):

| Path | Role |
|------|------|
| `data/prepared/` | Downloaded datasets as `*.joblib` |
| `data/embedded/<dataset>/<model>.(joblib\|json)` | Cached embeddings |
| `data/predictions/` | Head predictions |
| `data/meta.db` | Production results DB |
| `data/temp_dev.db` | **Dev / smoke-test** results DB |

## Two environments

### 1. Root project env — download + scoring

Used by `download.py` and `score.py`. Defined in root [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock).

Prefer **`uv run`** over `source …/activate` for the root env.

```bash
# Requires: uv, Python 3.11 (see .python-version)
uv sync
uv run python download.py ...
uv run python score.py ...
```

Python constraint: `>=3.11,<3.12`.

`install_deps.sh` / poetry / pyenv are **obsolete** for the root env.

### 2. Per-model env — embedding only

Each `model_wrappers/<model>/` has its own `venv` because dependencies conflict across models.

[`embed_wrapper.sh`](embed_wrapper.sh):

1. Reads `model_wrappers/<model>/.python-version` (default `3.11`)
2. Creates `venv` with `uv venv venv --seed --python=python$version` if missing
3. Installs [`base_requirements.txt`](base_requirements.txt) once via `uv pip`
4. Sources `init.sh` (model-specific installs, weights, `export HYDRA_EXPERIMENT=…`)
5. Runs `uv run --python <model>/venv/bin/python -u embed.py --multirun +experiment=$HYDRA_EXPERIMENT` plus any extra Hydra CLI args

```bash
./embed_wrapper.sh model_wrappers/<model>           # embed (uses cache)
./embed_wrapper.sh model_wrappers/<model> --no-cache
./embed_wrapper.sh model_wrappers/<model> -r         # recreate venv
./embed_wrapper.sh model_wrappers/<model> -d         # delete venv only

# Extra args after the wrapper path are forwarded to embed.py (Hydra overrides)
./embed_wrapper.sh model_wrappers/<model> \
  '+dataset=glob(clf_ogbg-molbbbp)'
```

## Hydra overview

Configs live under `config/`. Entry points:

| Script | `config_name` | Typical use |
|--------|---------------|-------------|
| `download.py` | `downloader` | Prepare datasets |
| `embed.py` | `embed` | Produce embeddings |
| `score.py` | `score` | Train/eval heads → SQLite |

### Composition pattern

- `+experiment=<name>` loads `config/experiment/<name>.yaml` (`# @package _global_`).
- Experiments often sweep models: `hydra.sweeper.params.+model: glob(fingerprints_*)`.
- Datasets are selected via `+dataset=…` / sweeper `+dataset: glob(…)`.
- Paths + DB file: `config/embedding/default.yaml` (`embedding.database`, directories).

CLI overrides **win** over values in `hydra.sweeper.params` (Hydra BasicSweeper). Prefer CLI or a dedicated experiment YAML over editing `embed.yaml` / `score.yaml`.

Useful references:

- [Hydra multi-run](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/)
- [Joblib launcher](https://hydra.cc/docs/1.3/plugins/joblib_launcher/)

### Cross-validated datasets (`cv_*`)

`config/dataset/cv_custom_clf_*.yaml` are CV twins of the `custom_clf_*` datasets: same source CSV (`source.filename`), but the shipped train/test holdout is replaced with 5 Butina-clustered folds (`cv.n_folds`, `cv.dist_threshold` — same procedure as `butina_split.py`). They carry a distinct `name` (`<dataset>_cv`), so they get their own `data/prepared/`, `data/embedded/`, and `data/predictions/` entries and never disturb the holdout results already in `data/meta.db`.

`Dataset.splits` / `EmbeddedDataset.splits` are therefore **either** a single `{split: [idx]}` dict **or** a list of them, one per fold. Use `dataset.is_multi_fold` / `dataset.n_folds` / `dataset.fold_splits(i)` rather than indexing `splits` directly; `get_train_data` / `get_test_data` take a `fold=` argument and raise if a CV dataset is used without one. Illegal-SMILES filtering and failed-embedding removal reindex every fold at once, so both shapes survive row drops.

```bash
uv run python download.py --multirun '+dataset=glob(cv_custom_*)'
./embed_wrapper.sh model_wrappers/<model> '+dataset=glob(cv_custom_*)'
```

`embed.yaml` / `score.yaml` sweep `glob(custom_*)`, which does **not** match `cv_custom_*` — CV runs need the explicit override above.

### Multirun + parallelism

Both embed and score configs set `hydra.mode: MULTIRUN` and sweep datasets (and sometimes models).

- **Default launcher** = serial local jobs.
- **Parallel local jobs**: override launcher to `joblib` and set `hydra.launcher.n_jobs` (see `config/score.yaml`, and optionally experiment configs like fingerprints).
- Scoring also uses scikit-learn `n_jobs` (default 32 in `src/eval/supervised/const.py`). Large datasets may set `memory_weight` so effective jobs ≈ `n_jobs / memory_weight`.

Example: parallel scoring with fewer workers:

```bash
uv run python score.py --multirun +experiment=fingerprints \
  hydra.launcher.n_jobs=4
```

### Common overrides

```bash
# Single dataset (BBBP) — preferred for development.
# NOTE: the `hydra.sweeper.params.+dataset=...` form does not parse (Hydra 1.3
# rejects `+` inside a dotted key, quoted or not); use the bare form below.
'+dataset=glob(clf_ogbg-molbbbp)'

# Temp results DB
++embedding.database=data/temp_dev.db

# Bypass embedding/score caches
++cache=false

# Score a subset of the supervised heads (score.py only; default is all of
# AVAILABLE_HEADS = rf, ridge, knn)
'+heads=[rf,ridge]'
```

## End-to-end: development smoke test (BBBP)

Goal: prove a model (or all of its variants) embeds + scores on BBBP and writes only to `data/temp_dev.db`.

### A. Prepare BBBP (root env)

```bash
uv sync
uv run python download.py --multirun \
  '+dataset=glob(clf_ogbg-molbbbp)'
```

### B. Embed BBBP (model env)

Pass Hydra overrides through `embed_wrapper.sh` (forwarded to `embed.py`):

```bash
./embed_wrapper.sh model_wrappers/<model> \
  '+dataset=glob(clf_ogbg-molbbbp)'
```

The wrapper sets `+experiment` from `HYDRA_EXPERIMENT` in that model’s `init.sh` (must match a file under `config/experiment/`).

Embeddings land under `data/embedded/ogbg-molbbbp/` (dataset `name` field inside the YAML, not the config filename).

### C. Score BBBP → temp DB (root env)

```bash
uv run python score.py --multirun \
  +experiment=<hydra_experiment_name> \
  '+dataset=glob(clf_ogbg-molbbbp)' \
  ++embedding.database=data/temp_dev.db
```

### D. Verify success

- Embedding file(s) exist under `data/embedded/ogbg-molbbbp/`.
- For multi-variant families, **every** swept model variant has an embedding and score rows.
- SQLite `data/temp_dev.db` has `ClassificationReport` rows for the embedder(s) on dataset `ogbg-molbbbp` (heads: `rf`, `ridge`, `knn` — see `AVAILABLE_HEADS`).
- No errors / incomplete multirun jobs in the Hydra output.

Optional check:

```bash
sqlite3 data/temp_dev.db \
  "SELECT embedder, model, test_metric FROM classificationreport WHERE dataset LIKE '%bbbp%';"
```

**Do not** report success until all variants in the experiment’s model sweep have completed BBBP embed + score.

## Adding a new model (agent checklist)

Follow [`docs/custom_model.md`](docs/custom_model.md). Minimal checklist:

1. **Create** `model_wrappers/<your_model>/` with:
   - `.python-version` — e.g. `3.11`
   - `wrapper.py` — subclass `SmilesEmbedder` or `GraphEmbedder` from `src/common/types.py`
     - `forward(...)` → `np.ndarray` shape `(N, D)`; failed rows may be `NaN` rows
     - `name` property — unique; must align with Hydra model config names
     - `get_embedder(name, **_kwargs)` factory (accept `**kwargs`; `task` is passed)
   - `init.sh` — install deps when `$INSTALL_DEP` is true; end with `export HYDRA_EXPERIMENT=<name>`
2. **Add** `config/experiment/<name>.yaml` (and `config/model/<name>_*.yaml` if multiple variants).
3. **Smoke-test** with BBBP + `data/temp_dev.db` as above.
4. If the experiment sweeps multiple models (`+model: glob(...)`), run the **full** sweep on BBBP before declaring done.
5. Only after smoke-test passes: full-benchmark runs (`./embed_wrapper.sh`, `./run_scoring.sh`) writing to `data/meta.db` are appropriate.

### Experiment config template

```yaml
# @package _global_
defaults:
  - _self_

model_name: your_model_name
```

Multi-variant (see `config/experiment/fingerprints.yaml`):

```yaml
# @package _global_
defaults:
  - _self_

hydra:
  sweeper:
    params:
      +model: glob(your_model_*)
```

### Full benchmark (post smoke-test)

```bash
./embed_wrapper.sh model_wrappers/<model>
# or background: ./run_embed.sh <model>

uv run python score.py --multirun +experiment=<name>
# or background: ./run_scoring.sh <name>
# default DB: data/meta.db
```

## Debugging tips

- **`embed.py` warns** it should not be called without the wrapper env — missing `wrapper` import usually means `PYTHONPATH` does not include `model_wrappers/<model>`.
- **Cache hits**: use `++cache=false` or `--no-cache` on the wrapper to force re-embed / re-score.
- **Illegal SMILES**: filtered via `config/illegal_smiles.txt` during embedding.
- **OOM / threading**: lower `hydra.launcher.n_jobs`; check dataset `memory_weight`; scoring sklearn `N_JOBS` is in `src/eval/supervised/const.py`.
- **Singleton DB context**: `DbContex` reads `embedding.database` relative to the original working directory (`hydra.utils.get_original_cwd()`).

## Out of scope for typical agent tasks

Paper figures (`visualizations.ipynb`), BBT / `bbt.ipynb`, and the Gradio `dashboard.py` are not required for adding or validating a model unless the user explicitly asks.
