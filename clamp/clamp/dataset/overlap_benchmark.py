"""
Build a PubChem cidx drop-set from overlap with every dataset used in this
repo's benchmark study, replacing clamp/dataset/overlap.py's original
MoleculeNet-only filter (8 hardcoded downstream sets).

Downstream molecules are sourced from this repo's data/prepared/*.joblib
cache (src.common.types.Dataset objects, produced by download.py from the
TDC/OGB configs in config/dataset/*.yaml) instead of CLAMP's own
setup_dataset()/MoleculeNet directory format, so no separate MoleculeNet
download is required.

example call (run from the repo root):
uv run python clamp/clamp/dataset/overlap_benchmark.py \
    --dset_path=./clamp/data/pubchem23/ \
    --prepared_dir=./data/prepared \
    --cidx_overlap_path=./clamp/data/pubchem23/cidx_overlap_benchmark.npy
"""
import sys
from pathlib import Path

# data/prepared/*.joblib files are pickled src.common.types.Dataset objects;
# unpickling requires that module importable, but this script lives inside
# the clamp/ package tree, so the repo root has to be added explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, RDLogger
from tqdm.contrib.concurrent import process_map

from clamp.dataset.overlap import compute_inchii
from src.common.types import Dataset  # noqa: F401  (needed to unpickle data/prepared/*.joblib)

RDLogger.DisableLog('rdApp.*')


def default_njobs(fraction: float = 0.5) -> int:
    """Cap parallel workers to a fraction of available CPUs (default: 50%)."""
    total = os.cpu_count() or 1
    return max(1, int(total * fraction))


def load_pubchem_with_inchii(dset_path: Path, standardize_smiles: bool, njobs: int) -> pd.DataFrame:
    cache_path = dset_path / 'compound_smiles_with_inchii.parquet'
    if cache_path.exists():
        logger.info(f'loading cached pubchem inchii from {cache_path}')
        return pd.read_parquet(cache_path)

    df = pd.read_parquet(dset_path / 'compound_smiles.parquet')
    df = compute_inchii(df, save_df=False, standardize_smiles=standardize_smiles, njobs=njobs)
    df.to_parquet(cache_path)
    return df


def raw_inchikey(smiles):
    """InChIKey of the molecule exactly as written (no standardization), or None."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol) or None
    except Exception:
        return None


def raw_inchikeys(smiles: list, njobs: int) -> list:
    return process_map(raw_inchikey, smiles, max_workers=njobs, chunksize=1000,
                       desc='raw inchikeys')


def load_pubchem_raw_inchikeys(dset_path: Path, njobs: int) -> pd.Series:
    cache_path = dset_path / 'compound_inchikey_raw.parquet'
    if cache_path.exists():
        logger.info(f'loading cached pubchem raw inchikeys from {cache_path}')
        return pd.read_parquet(cache_path)['inchikey']

    df = pd.read_parquet(dset_path / 'compound_smiles.parquet')
    df['inchikey'] = raw_inchikeys(df.CanonicalSMILES.astype(str).tolist(), njobs)
    df[['CID', 'inchikey']].to_parquet(cache_path)
    return df['inchikey']


def select_datasets(prepared_dir: Path, datasets_file: Optional[Path]) -> list:
    joblib_files = sorted(prepared_dir.glob('*.joblib'))
    if not joblib_files:
        raise FileNotFoundError(f'No .joblib files found in {prepared_dir}')
    if datasets_file is None:
        return joblib_files

    wanted = [l.strip() for l in datasets_file.read_text().splitlines()
              if l.strip() and not l.lstrip().startswith('#')]
    by_stem = {p.stem: p for p in joblib_files}
    missing = [w for w in wanted if w not in by_stem]
    if missing:
        raise FileNotFoundError(f'{missing} listed in {datasets_file} but not in {prepared_dir}')
    return [by_stem[w] for w in wanted]


def build_overlap(dset_path: Path, prepared_dir: Path, standardize_smiles: bool, njobs: int,
                  splits: str = 'all', datasets_file: Optional[Path] = None,
                  match_raw_inchikey: bool = False) -> set:
    """splits='all' matches every molecule of each dataset; splits='test' only its
    test split (the criterion the CLAMP paper used, Appendix A.2.4).

    match_raw_inchikey additionally drops PubChem compounds whose unstandardized
    InChIKey equals a dataset molecule's. The standardized-InChI match alone is
    input-representation-sensitive (tautomer canonicalization depends on how the
    SMILES is written), so it misses some exact duplicates."""
    pubchem_mols_df = load_pubchem_with_inchii(dset_path, standardize_smiles, njobs)
    pubchem_keys = load_pubchem_raw_inchikeys(dset_path, njobs) if match_raw_inchikey else None

    joblib_files = select_datasets(prepared_dir, datasets_file)

    all_idxs = set()
    for path in joblib_files:
        dataset = joblib.load(path)
        data = dataset.data
        if splits == 'test':
            test_idx = list(dataset.splits.get('test', []))
            if not test_idx:
                logger.warning(f'{dataset.name}: no test split, skipping')
                continue
            data = data.iloc[test_idx]
        smiles = data['smiles'].dropna().astype(str)

        ds_df = pd.DataFrame({'CanonicalSMILES': smiles.values})
        ds_df = compute_inchii(ds_df, standardize_smiles=standardize_smiles, njobs=njobs)
        # standardize() returns None on unparseable SMILES, which smiles2inchii
        # turns into the literal string 'None' on both sides; matching on it
        # would drop every unparseable PubChem compound for no reason.
        inchii_set = set(ds_df.inchii) - {'None'}

        drop_mask = pubchem_mols_df.inchii.isin(inchii_set)
        if match_raw_inchikey:
            key_set = {k for k in raw_inchikeys(smiles.tolist(), njobs) if k}
            raw_mask = pubchem_keys.isin(key_set).values
            logger.info(f'{dataset.name}: raw inchikey match adds '
                        f'{int((raw_mask & ~drop_mask.values).sum())} compounds')
            drop_mask = drop_mask | raw_mask
        drop_idxs = pubchem_mols_df.index[drop_mask]

        logger.info(
            f'{dataset.name}: {len(drop_idxs)} overlapping pubchem compounds '
            f'({drop_mask.mean() * 100:.4f}% of pubchem), from {len(ds_df)} molecules in dataset'
        )
        all_idxs.update(drop_idxs.tolist())

    return all_idxs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build a PubChem cidx drop-set from overlap with all benchmark-study "
                    "datasets (this repo's data/prepared/*.joblib cache)."
    )
    parser.add_argument('--dset_path', type=str, default='./clamp/data/pubchem23/')
    parser.add_argument('--prepared_dir', type=str, default='./data/prepared',
                        help="Path to this repo's data/prepared/ joblib cache.")
    parser.add_argument('--cidx_overlap_path', type=str,
                        default='./clamp/data/pubchem23/cidx_overlap_benchmark.npy')
    parser.add_argument('--standardize_smiles', type=bool, default=True)
    parser.add_argument('--splits', choices=['all', 'test'], default='all',
                        help="'all' drops every benchmark molecule; 'test' only test-split "
                             "molecules (the paper's own criterion).")
    parser.add_argument('--njobs', type=int, default=default_njobs(),
                        help='Worker processes for InChI conversion (default: 50%% of CPUs).')
    parser.add_argument('--datasets_file', type=str, default=None,
                        help='Text file of dataset names (joblib stems, one per line) to use; '
                             'default: every *.joblib in --prepared_dir.')
    parser.add_argument('--match_raw_inchikey', action='store_true',
                        help='Also drop PubChem compounds whose unstandardized InChIKey matches.')
    args = parser.parse_args()

    dset_path = Path(args.dset_path)
    prepared_dir = Path(args.prepared_dir)
    datasets_file = Path(args.datasets_file) if args.datasets_file else None

    all_idxs = build_overlap(dset_path, prepared_dir, args.standardize_smiles, args.njobs,
                             splits=args.splits, datasets_file=datasets_file,
                             match_raw_inchikey=args.match_raw_inchikey)

    logger.info(f'found {len(all_idxs)} pubchem cidx to drop across '
                f'{len(select_datasets(prepared_dir, datasets_file))} benchmark datasets')
    logger.info(f'saving to {args.cidx_overlap_path}')
    np.save(args.cidx_overlap_path, np.array(sorted(all_idxs)))
