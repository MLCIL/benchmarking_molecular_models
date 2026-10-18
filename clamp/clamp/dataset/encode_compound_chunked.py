"""
Memory-safe variant of encode_compound.py for very large compound sets
(e.g. PubChem23's 3.47M compounds).

encode_compound.py's dense fp_types (e.g. morganc+rdkc) return a float64
array of length fp_size per molecule. For PubChem23 (fp_size=8096) that's
a ~225GB feature matrix; building it the way encode_compound.py does --
a Python list of per-molecule arrays, then np.array(list) -- holds both
the list and the freshly materialized array in memory at once (~450GB
peak), which OOMs on a shared machine.

This script produces the exact same output (compound_features_{fp_type}.npy,
same path convention as encode_compound.py) by streaming batches directly
into a preallocated memmapped .npy file, so peak RAM stays at roughly one
batch's size regardless of total dataset size. Per-molecule fingerprint
computation itself (mhnreact.molutils._getFingerprint) is unchanged, so
output values are identical to encode_compound.py's.

example call (run from the repo root):
uv run python clamp/clamp/dataset/encode_compound_chunked.py \
    --compounds=./clamp/data/pubchem23/compound_names.parquet \
    --compound2smiles=./clamp/data/pubchem23/compound_smiles.parquet \
    --fp_type=morganc+rdkc --fp_size=8096 --njobs=64 --batch_size=50000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from mhnreact.molutils import _getFingerprint, disable_rdkit_logging
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

disable_rdkit_logging()


def encode_chunked(list_of_smiles, fp_size, radius, which, njobs, batch_size, out_path,
                   dtype=np.float32):
    n = len(list_of_smiles)
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=dtype, shape=(n, fp_size))

    for start in tqdm(range(0, n, batch_size), desc='encoding in batches'):
        end = min(start + batch_size, n)
        inp = [(smi, fp_size, radius, False, which) for smi in list_of_smiles[start:end]]
        fps = process_map(_getFingerprint, inp, max_workers=njobs, chunksize=1, disable=True)
        out[start:end] = np.array(fps, dtype=dtype)
        out.flush()

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Compute RDKit sparse features for a collection of PubChem compounds (chunked, memory-safe).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--compound2smiles', help='Path to a Parquet file mapping CIDs to SMILES strings.')
    parser.add_argument('--compounds', default=None, help='Path to a Parquet file with compound ids (CIDs) for which to extract features. If none is provided a 1:1 mapping is assumed.')
    parser.add_argument('--fp_type', help='fp_type, e.g. morganc+rdkc, MxFP', default='morganc+rdkc', type=str)
    parser.add_argument('--fp_size', help='fp_size', default=8096, type=int)
    parser.add_argument('--radius', help='morgan radius', default=2, type=int)
    parser.add_argument('--njobs', help='njobs', default=50, type=int)
    parser.add_argument('--batch_size', help='number of molecules encoded per batch', default=50000, type=int)
    parser.add_argument('--smiles_column', help='smiles_column', default='CanonicalSMILES', type=str)
    parser.add_argument('--dtype', help='output dtype; CLAMP models are float32 and the dataloader '
                                        'does not cast, so float64 will fail at training time',
                        default='float32', choices=['float32', 'float64'], type=str)
    args = parser.parse_args()

    compound2smiles_df = pd.read_parquet(args.compound2smiles)
    if 'CID' not in compound2smiles_df.columns:
        logger.warning('CID not in compound2smiles_df.columns, using the index as CID.')
        compound2smiles_df['CID'] = compound2smiles_df.index
    compound2smiles = compound2smiles_df.set_index('CID')[args.smiles_column].to_dict()

    if args.compounds is None:
        compounds = list(range(len(compound2smiles)))
        logger.info('No compound CID to idx provided, assuming 1:1 mapping.')
    else:
        compounds_df = pd.read_parquet(args.compounds)
        compounds = compounds_df['CID'].squeeze().tolist()

    list_of_smiles = [compound2smiles[c] for c in compounds]
    logger.info(f'converting {len(list_of_smiles)} smiles to features (batch_size={args.batch_size})')

    out_path = Path(args.compound2smiles).with_name(f'compound_features_{args.fp_type}.npy')
    out = encode_chunked(list_of_smiles, args.fp_size, args.radius, args.fp_type, args.njobs,
                         args.batch_size, out_path, dtype=np.dtype(args.dtype))
    logger.info(f'saved compound features with shape {out.shape} dtype {out.dtype} to {out_path}')
