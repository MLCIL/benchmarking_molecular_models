#!/bin/bash
# CLAMP retrain: pretrain CLAMP from scratch on PubChem18 with every benchmark
# molecule removed, then validate the result.
#
# This is a reproduction of the released retrained model:
#   https://huggingface.co/scikit-fingerprints/benchmarking_models_clamp_refitted
#
# Hyperparameters: the original authors' hp.json (hparams/authors_hp.json,
# verbatim; model=ScaledMLPLayerNorm, i.e. a learned temperature) with two changes:
#   - train_only_actives off: the paper's Eq. 2 is BCE over both classes, and
#     dropping the inactives collapses training..
#
# Overlap filter: every PubChem18 compound that occurs in any split of the 25
# primary benchmark datasets (benchmark_datasets_primary25.txt), matched on the
# standardized InChI and on the raw InChIKey (the standardized match alone
# misses exact duplicates). An independent exact-identity check (InChIKey and
# canonical SMILES) must find zero leaks before training starts.
#
# Inputs: PubChem18 (the paper's pretraining dataset) is downloaded, then its
# compound features (morganc+rdkc, 8192 bits, float32) and assay features (CLIP
# over assay titles) are built. The 25 benchmark datasets must already be
# prepared in data/prepared.
#
# Every stage is idempotent: existing outputs are reused unless --force.
# Fingerprinting takes ~35 min; training runs on the full ~1e8 filtered rows
# per epoch, so expect a day or more.
#
# Usage (from anywhere):
#   ./clamp/retrain_clamp_noleak.sh                # full run, reuse what exists
#   ./clamp/retrain_clamp_noleak.sh --check-only   # verify artifacts, no compute
#   NJOBS=102 GPU=1 ./clamp/retrain_clamp_noleak.sh
#
# Background it:  nohup ./clamp/retrain_clamp_noleak.sh > clamp/retrain.log 2>&1 &

set -euo pipefail

# Run from the repo root (mlflow.db + mlruns/ live there), with clamp's own venv.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/clamp/.venv/bin/python"
export PYTHONPATH="$REPO/clamp${PYTHONPATH:+:$PYTHONPATH}"
export MLFLOW_TRACKING_URI="sqlite:///$REPO/mlflow.db"

PC18="./clamp/data/pubchem18"
PREPARED="./data/prepared"
DATASETS="./clamp/benchmark_datasets_primary25.txt"
FILTER="$PC18/cidx_overlap_primary25_v2.npy"
HP_AUTHORS="./clamp/hparams/authors_hp.json"
HP_RUN="./clamp/hparams/authors_hp_v3.json"
HP_URL="https://cloud.ml.jku.at/s/dRX9TWPrF7WqnHd/download/hp.json"
EXPERIMENT="clamp_noleak"
WEIGHTS="./model_wrappers/clamp/weights/clamp_noleak"

NJOBS="${NJOBS:-$(( $(nproc) * 4 / 5 ))}"
GPU="${GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NJOBS}"
# Pin the physical GPU (nvidia-smi numbering); inside, it is always cuda:0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

FORCE=false; CHECK_ONLY=false
for a in "$@"; do case "$a" in
    --force) FORCE=true ;;
    --check-only) CHECK_ONLY=true ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
esac; done

PASS=0; FAIL=0
step() { echo; echo "=== $(date '+%F %T') $* ==="; }
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
assert_eq() { if [ "$1" = "$2" ]; then ok "$3: $1"; else bad "$3: got '$1', expected '$2'"; fi; }
have() { [ -s "$1" ] && [ "$FORCE" = false ]; }
skip_compute() { [ "$CHECK_ONLY" = true ]; }
die_on_fail() { [ "$FAIL" -eq 0 ] || { echo "  aborting: $FAIL check(s) failed" >&2; exit 1; }; }
py()  { "$PY" "$@" 2>&1 | grep -viE "futurewarning|warnings\.warn|^\s*$" || true; }
run() { nice -n 10 "$PY" "$@"; }

# ---------------------------------------------------------------------------
step "0. Preflight"
[ -x "$PY" ] || { echo "missing $PY (create clamp/.venv first)" >&2; exit 1; }
assert_eq "$(grep -vc '^#' "$DATASETS")" "25" "datasets in $DATASETS"
assert_eq "$(grep -v '^#' "$DATASETS" | while read -r d; do [ -s "$PREPARED/$d.joblib" ] && echo "$d"; done | wc -l)" \
    "25" "listed datasets prepared in $PREPARED"
assert_eq "$(grep -c 'dtype=np\.bool\b' clamp/clamp/utils.py || true)" "0" "no removed np.bool aliases"
die_on_fail

# ---------------------------------------------------------------------------
step "1. PubChem18 (paper Table A3: 2,120,811 compounds / 21,002 assays)"
if have "$PC18/activity.parquet"; then
    echo "  present, reusing"
elif ! skip_compute; then
    mkdir -p ./clamp/data && pushd ./clamp/data >/dev/null
    wget -q -O pubchem18.zip https://cloud.ml.jku.at/s/2ybfLRXWSYb4DZN/download
    unzip -q pubchem18.zip && rm pubchem18.zip
    [ -d data/pubchem18 ] && mv data/pubchem18 . && rmdir data || true   # archive nests ./data/
    popd >/dev/null
fi
assert_eq "$(py -c "import pandas as pd; print(len(pd.read_parquet('$PC18/compound_smiles.parquet')))" | tail -1)" \
    "2120811" "PubChem18 compounds"
assert_eq "$(py -c "import pandas as pd; print(len(pd.read_parquet('$PC18/assay_names.parquet')))" | tail -1)" \
    "21002" "PubChem18 assays"
die_on_fail

# ---------------------------------------------------------------------------
step "2. Compound features (morganc+rdkc, fp_size=8192, float32)"
# float32 is mandatory: dataloader._load_compound does no dtype cast, so float64
# features crash the model with "mat1 and mat2 must have the same dtype".
if have "$PC18/compound_features_morganc+rdkc.npy"; then
    echo "  present, reusing"
elif ! skip_compute; then
    run clamp/clamp/dataset/encode_compound_chunked.py \
        --compounds="$PC18/compound_names.parquet" \
        --compound2smiles="$PC18/compound_smiles.parquet" \
        --fp_type=morganc+rdkc --fp_size=8192 --njobs="$NJOBS" --dtype=float32
fi
assert_eq "$(py -c "
import numpy as np; a=np.load('$PC18/compound_features_morganc+rdkc.npy', mmap_mode='r'); print(f'{a.shape}|{a.dtype}')" | tail -1)" \
    "(2120811, 8192)|float32" "compound features shape|dtype"
die_on_fail

# ---------------------------------------------------------------------------
step "3. Assay features (CLIP over assay titles)"
if have "$PC18/assay_features_clip.npy"; then
    echo "  present, reusing"
elif ! skip_compute; then
    run clamp/clamp/dataset/encode_assay.py \
        --assay_path="$PC18/assay_names.parquet" --encoding=clip --gpu=0 --columns title
fi
assert_eq "$(py -c "
import numpy as np; print(np.load('$PC18/assay_features_clip.npy', mmap_mode='r').shape)" | tail -1)" \
    "(21002, 512)" "assay features shape"
die_on_fail

# ---------------------------------------------------------------------------
step "4. Hyperparameters: authors' hp.json, train_only_actives off (ScaledMLPLayerNorm as in hp.json)"
if ! skip_compute; then
    curl -sSfL -o "$REPO/clamp/hparams/.authors_hp.download" "$HP_URL" \
        && { cmp -s "$REPO/clamp/hparams/.authors_hp.download" "$HP_AUTHORS" \
             && ok "$HP_AUTHORS identical to $HP_URL" \
             || bad "$HP_AUTHORS differs from $HP_URL"; } \
        || echo "  (could not download $HP_URL; skipping provenance check)"
    rm -f "$REPO/clamp/hparams/.authors_hp.download"
fi
# Every model/optimizer hyperparameter train.py resolves from $HP_RUN must
# equal the authors'. Keys that are CLI flags (dataset, gpu, wandb, ...) are
# never read from the json by train.py, so they are passed explicitly below.
assert_eq "$(py - <<PYEOF | tail -1
import sys, json
sys.argv = ['train.py', '--hyperparams=$HP_RUN', '--split=time_a', '--assay_mode=clip',
            '--compound_mode=morganc+rdkc']
from clamp.train import parse_args_override
a = vars(parse_args_override())
hp = {k: (None if v == 'None' else v) for k, v in json.load(open('$HP_AUTHORS')).items()}
bookkeeping = {'dataset', 'drop_cidx_path', 'experiment', 'gpu', 'hyperparams', 'checkpoint',
               'random', 'verbose', 'wandb', 'bf16'}
intended = {'train_only_actives': False, 'train_subsample': 0, 'model': 'ScaledMLPLayerNorm'}
diff = sorted(k for k in a if k not in bookkeeping and k not in intended and a[k] != hp.get(k, '<absent>'))
wrong = sorted(k for k, v in intended.items() if a[k] != v)
print('ok' if not diff and not wrong else f'diff={diff} wrong={wrong}')
PYEOF
)" "ok" "resolved hparams == authors' except train_only_actives=False, train_subsample=0"
die_on_fail

# ---------------------------------------------------------------------------
step "5. Overlap filter (25 primary datasets, all splits, std InChI + raw InChIKey)"
if ! have "$FILTER" && ! skip_compute; then
    run clamp/clamp/dataset/overlap_benchmark.py --dset_path="$PC18" --prepared_dir="$PREPARED" \
        --datasets_file="$DATASETS" --splits=all --match_raw_inchikey \
        --cidx_overlap_path="$FILTER" --njobs="$NJOBS"
fi
FILTER_N=$(py -c "import numpy as np; print(len(np.load('$FILTER', allow_pickle=True)))" | tail -1)
echo "  filter size: $FILTER_N PubChem18 compounds"
assert_eq "$(py -c "
import numpy as np; a=np.load('$FILTER', allow_pickle=True)
print('ok' if len(a) and a.min() >= 0 and a.max() < 2120811 else 'out-of-range')" | tail -1)" "ok" \
    "filter indices within PubChem18"

# ---------------------------------------------------------------------------
step "6. Independent leak check (exact InChIKey + canonical SMILES) must be zero"
LEAKS_CSV="$PC18/leaks_primary25_v2.csv"
if ! skip_compute && ! have "$PC18/leaks_primary25_v2.txt"; then
    run clamp/clamp/dataset/check_benchmark_leaks.py --dset_path="$PC18" --prepared_dir="$PREPARED" \
        --datasets_file="$DATASETS" --drop_cidx_path="$FILTER" --out_csv="$LEAKS_CSV" \
        --njobs="$NJOBS" 2>/dev/null | grep '^LEAKS' > "$PC18/leaks_primary25_v2.txt" || true
fi
cat "$PC18/leaks_primary25_v2.txt"
assert_eq "$(grep -c 'train=0 valid=0' "$PC18/leaks_primary25_v2.txt")" "2" "no benchmark molecule left in train/valid"
die_on_fail

# ---------------------------------------------------------------------------
step "7. Pretraining on physical GPU $GPU"
# No --train_only_actives (the paper's Eq.2 is BCE over both classes; dropping
# inactives collapses training) and train_subsample=0 (full training set).
run_id() {
    "$PY" -c "
import mlflow; r = mlflow.search_runs(experiment_names=['$EXPERIMENT'], filter_string=\"attributes.status = 'FINISHED'\",
                                      order_by=['attributes.start_time DESC'], max_results=1)
print(r.run_id.iloc[0] if len(r) else '')" 2>/dev/null | tail -1
}
RUN_ID=$(run_id)
if [ -n "$RUN_ID" ] && [ "$FORCE" = false ]; then
    echo "  finished run $RUN_ID exists in experiment $EXPERIMENT, reusing"
elif ! skip_compute; then
    run clamp/clamp/train.py --dataset="$PC18/" --hyperparams="$HP_RUN" \
        --split=time_a --assay_mode=clip --compound_mode=morganc+rdkc \
        --drop_cidx_path="$FILTER" --gpu=0 --experiment="$EXPERIMENT"
    RUN_ID=$(run_id)
fi
[ -n "$RUN_ID" ] && ok "MLflow run $RUN_ID" || { bad "no finished run in experiment $EXPERIMENT"; die_on_fail; }

# ---------------------------------------------------------------------------
step "8. Stage weights for the model wrapper"
CKPT_SRC=$("$PY" -c "
import mlflow; print(mlflow.get_run('$RUN_ID').info.artifact_uri.replace('file://', ''))" 2>/dev/null | tail -1)/checkpoint.pt
[ -s "$CKPT_SRC" ] || { bad "missing $CKPT_SRC"; die_on_fail; }
if ! have "$WEIGHTS/checkpoint.pt" && ! skip_compute; then
    mkdir -p "$WEIGHTS" && cp "$CKPT_SRC" "$WEIGHTS/checkpoint.pt"
fi
assert_eq "$(md5sum < "$WEIGHTS/checkpoint.pt" | cut -c1-32)" "$(md5sum < "$CKPT_SRC" | cut -c1-32)" \
    "staged checkpoint == run $RUN_ID"
# Architecture description consumed by model_wrappers/clamp/wrapper.py.
"$PY" - <<PYEOF
import json
hp = json.load(open('$HP_RUN'))
arch = {k: hp[k] for k in ('model', 'compound_features_size', 'assay_features_size', 'embedding_size',
                           'hidden_layers', 'compound_layer_sizes', 'assay_layer_sizes', 'dropout_input',
                           'dropout_hidden', 'nonlinearity', 'norm', 'compound_mode', 'assay_mode',
                           'split', 'lr_ini', 'batch_size', 'optimizer', 'loss_fun', 'l2')}
arch.update(dataset='$PC18/', drop_cidx_path='$FILTER', mlflow_run_id='$RUN_ID')
json.dump(arch, open('$WEIGHTS/hp.json', 'w'), indent=2)
PYEOF
ok "wrote $WEIGHTS/hp.json"

# ---------------------------------------------------------------------------
step "9. Verify the retrained CLAMP loads (strict) and embeds through the wrapper"
RES=$(PYTHONPATH=.:model_wrappers/clamp TORCH_GPU="cuda:0" \
      model_wrappers/clamp/venv/bin/python - <<'PYEOF' 2>&1 | tail -1
import numpy as np
from wrapper import get_embedder
emb = get_embedder("CLAMP_noleak")
out = emb.forward(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
print(f"{emb.name}|{out.shape[1]}|{bool(np.isfinite(out).all())}")
PYEOF
)
assert_eq "$RES" "CLAMP_noleak|768|True" "wrapper embeds (name|dim|finite)"
# The original authors' architecture plus exactly one learned temperature:
# keys/shapes == the authors' CLAMP checkpoint's + {'scale': (1,)}.
assert_eq "$("$PY" -c "
import torch
a = torch.load('$WEIGHTS/checkpoint.pt', map_location='cpu', weights_only=False)['model_state_dict']
b = torch.load('./model_wrappers/clamp/weights/clamp_clip/checkpoint.pt', map_location='cpu', weights_only=False)['model_state_dict']
print('same' if {k: tuple(v.shape) for k, v in a.items()} == {**{k: tuple(v.shape) for k, v in b.items()}, 'scale': (1,)} else 'differs')" 2>/dev/null | tail -1)" \
    "same" "state-dict keys/shapes == authors' CLAMP checkpoint + scale"

# ---------------------------------------------------------------------------
step "10. time_a test metrics (best-valid checkpoint)"
"$PY" - <<PYEOF 2>/dev/null
import mlflow
r = mlflow.get_run('$RUN_ID').data
for k in ('test_mean_davgp', 'test_mean_auroc', 'test_mean_avgp', 'valid_mean_davgp'):
    print(f'  {k:18s} {r.metrics.get(k, float("nan")):.4f}')
PYEOF

echo
echo "==================== SUMMARY ===================="
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ] && echo "  CLAMP RETRAIN VALIDATED" || echo "  VALIDATION FAILED"
exit $(( FAIL > 0 ))
