"""
Independent post-filter leak check for a benchmark overlap filter.

Takes the compounds that actually remain in CLAMP's train and valid splits once
the --drop_cidx_path filter is applied (same split arithmetic as
InMemoryClamp._find_splits / setup_dataset for split=time_a) and looks for
benchmark molecules among them by two exact-identity keys that do not go
through clamp.mol_utils.standardize:

- raw InChIKey of the SMILES as written
- RDKit canonical isomeric SMILES

Prints one line per key, `LEAKS <key> train=<n> valid=<n>`, and writes every
hit to --out_csv. Exit code is 1 if anything leaks.

example call (run from the repo root):
clamp/.venv/bin/python clamp/clamp/dataset/check_benchmark_leaks.py \
    --dset_path=./clamp/data/pubchem18/ --prepared_dir=./data/prepared \
    --datasets_file=./clamp/benchmark_datasets_primary25.txt \
    --drop_cidx_path=./clamp/data/pubchem18/cidx_overlap_primary25_v2.npy
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, RDLogger
from tqdm.contrib.concurrent import process_map

from clamp.dataset.overlap_benchmark import default_njobs, select_datasets
from src.common.types import Dataset  # noqa: F401  (needed to unpickle data/prepared/*.joblib)

RDLogger.DisableLog('rdApp.*')


def identity_keys(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None
        return Chem.MolToInchiKey(mol) or None, Chem.MolToSmiles(mol)
    except Exception:
        return None, None


def keys(smiles: list, njobs: int):
    out = process_map(identity_keys, smiles, max_workers=njobs, chunksize=1000, desc='identity keys')
    return [k for k, _ in out], [s for _, s in out]


def time_a_compounds(dset_path: Path, drop: set, train_size: float = 0.6):
    act = pd.read_parquet(dset_path / 'activity.parquet', columns=['compound_idx', 'assay_idx'])
    n_assays = len(pd.read_parquet(dset_path / 'assay_names.parquet'))
    # InMemoryClamp._chunk
    first_cut = int(round(train_size * n_assays))
    second_cut = first_cut + int(round((n_assays - first_cut) / 2))
    a, c = act.assay_idx.values, act.compound_idx.values
    train = set(np.unique(c[a <= first_cut - 1]).tolist()) - drop
    valid = set(np.unique(c[(a >= first_cut) & (a <= second_cut - 1)]).tolist()) - drop
    return train, valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dset_path', type=str, default='./clamp/data/pubchem18/')
    parser.add_argument('--prepared_dir', type=str, default='./data/prepared')
    parser.add_argument('--datasets_file', type=str, default=None)
    parser.add_argument('--drop_cidx_path', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default=None)
    parser.add_argument('--njobs', type=int, default=default_njobs())
    args = parser.parse_args()

    dset_path = Path(args.dset_path)
    files = select_datasets(Path(args.prepared_dir), Path(args.datasets_file) if args.datasets_file else None)

    bench = {'inchikey': {}, 'smiles': {}}
    for path in files:
        ds = joblib.load(path)
        ik, cs = keys(ds.data['smiles'].dropna().astype(str).tolist(), args.njobs)
        for name, ks in (('inchikey', ik), ('smiles', cs)):
            for k in ks:
                if k:
                    bench[name].setdefault(k, set()).add(ds.name)
    logger.info(f'{len(files)} datasets, {len(bench["inchikey"])} unique benchmark InChIKeys')

    drop = set(np.load(args.drop_cidx_path, allow_pickle=True).tolist())
    train, valid = time_a_compounds(dset_path, drop)
    logger.info(f'compounds remaining after filter: train {len(train)}, valid {len(valid)}')

    used = sorted(train | valid)
    pubchem_smiles = pd.read_parquet(dset_path / 'compound_smiles.parquet').CanonicalSMILES.values
    ik, cs = keys([pubchem_smiles[i] for i in used], args.njobs)

    hits = []
    for name, ks in (('inchikey', ik), ('smiles', cs)):
        n_train = n_valid = 0
        for cidx, k in zip(used, ks):
            if k in bench[name]:
                n_train += cidx in train
                n_valid += cidx in valid
                hits.append((name, cidx, k, pubchem_smiles[cidx], cidx in train, cidx in valid,
                             ','.join(sorted(bench[name][k]))))
        print(f'LEAKS {name} train={n_train} valid={n_valid}')

    if args.out_csv:
        pd.DataFrame(hits, columns=['key', 'cidx', 'value', 'pubchem_smiles', 'in_train', 'in_valid',
                                    'datasets']).to_csv(args.out_csv, index=False)
    sys.exit(1 if hits else 0)
