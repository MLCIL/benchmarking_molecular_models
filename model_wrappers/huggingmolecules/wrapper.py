from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import hashlib
import multiprocessing as mp
import os

import joblib
import torch
import logging as log
from rdkit import Chem
from tqdm import tqdm

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device, get_embed_cpu_njobs
from huggingmolecules import MatFeaturizer, RMatFeaturizer, GroverFeaturizer, MatModel, RMatModel, GroverModel
from huggingmolecules.featurization import featurization_mat, featurization_rmat
from huggingmolecules.featurization.featurization_mat_utils import get_mol_from_smiles


# MAT / R-MAT featurization calls get_mol_from_smiles, which runs RDKit
# EmbedMolecule(maxAttempts=5000) + UFF serially per molecule. On HIV a few
# organometallics fail every attempt and burn minutes each before the 2D
# fallback, which made MUV/HIV embedding infeasible. We run the *unchanged*
# get_mol_from_smiles in a process pool once per dataset, keep the resulting
# mols in memory and on disk, and let the MAT/R-MAT featurizers read from that
# cache, so all MAT variants share one conformer per molecule.
CONFORMER_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'conformers', 'huggingmolecules')
_conformers = {}  # smiles -> RDKit mol binary, or the error message get_mol_from_smiles raised


class ConformerGenerationError(RuntimeError):
    pass


def _conformer_job(smiles):
    try:
        return smiles, get_mol_from_smiles(smiles).ToBinary()
    except Exception as e:
        return smiles, f'{type(e).__name__}: {e}'


def _cached_get_mol_from_smiles(smiles):
    hit = _conformers.get(smiles)
    if hit is None:
        return get_mol_from_smiles(smiles)
    if isinstance(hit, str):
        raise ConformerGenerationError(f'Conformer generation failed: {hit}')
    return Chem.Mol(hit)


featurization_mat.get_mol_from_smiles = _cached_get_mol_from_smiles
featurization_rmat.get_mol_from_smiles = _cached_get_mol_from_smiles


def _ensure_conformers(smiles):
    unique = list(dict.fromkeys(smiles))
    if all(s in _conformers for s in unique):
        return
    path = os.path.join(CONFORMER_CACHE_DIR, hashlib.sha1('\n'.join(unique).encode()).hexdigest()[:16] + '.joblib')
    if os.path.exists(path):
        log.info(f'Loading conformer cache {path}')
        _conformers.update(joblib.load(path))
    missing = [s for s in unique if s not in _conformers]
    if not missing:
        return

    n_jobs = get_embed_cpu_njobs()
    log.info(f'Generating conformers for {len(missing)} molecules with {n_jobs} workers')
    # fork: workers only run RDKit, never CUDA
    with mp.get_context('fork').Pool(n_jobs) as pool:
        for s, mol in tqdm(pool.imap_unordered(_conformer_job, missing), total=len(missing)):
            _conformers[s] = mol
    failed = [s for s in missing if isinstance(_conformers[s], str)]
    log.info(f'Conformer generation failed for {len(failed)} molecules (embedded as zero vectors): {failed}')

    os.makedirs(CONFORMER_CACHE_DIR, exist_ok=True)
    joblib.dump({s: _conformers[s] for s in unique}, path)
    log.info(f'Saved conformer cache {path}')


class HuggingMoleculesEmbedder(SmilesEmbedder):
    FEAT_CLS = None
    MODEL_CLS = None
    OUTPUT_DIM = None
    NEEDS_CONFORMERS = False

    def __init__(self, model_path: str):
        self._model_name = model_path
        self._device = get_device()
        self._featurizer = self.FEAT_CLS.from_pretrained(model_path)
        self._model = self.MODEL_CLS.from_pretrained(model_path).to(self._device)
        self._batch_size = 128

    def process_step(self, single_smile):
        return self.process_batch([single_smile])

    def process_batch(self, smiles):
        batch = self._featurizer(smiles).to(self._device)
        with torch.no_grad():
            output = self._model(batch) # Returns Torch Tensor of shape [n_samples, dims]
        return output

    def safe_batch(self, smiles):
        try:
            return self.process_batch(smiles)
        except Exception as e:
            log.warning(f"Batch failed, processing each smile individually: {e}")
            batch_values = []
            for smile in smiles:
                try:
                    batch_values.append(self.process_step(smile))
                except Exception as e:
                    log.error(f"Failed to process smile '{smile}': {e}")
                    # No conformer means MAT/R-MAT have nothing to embed; a real pipeline
                    # would still have to score the molecule, so emit a zero vector rather
                    # than dropping it. Any other failure stays NaN and is dropped.
                    fill = 0.0 if isinstance(e, ConformerGenerationError) else float('nan')
                    err_array = torch.full((1, self.OUTPUT_DIM), fill, device=self._device)

                    batch_values.append(err_array)
            return torch.cat(batch_values, dim=0)

    def forward(self, smiles):
        if self.NEEDS_CONFORMERS:
            _ensure_conformers(list(smiles))
        return torch.cat(
            [self.safe_batch(b).detach().cpu() for b in batch(smiles, self._batch_size)],
            dim=0
        ).numpy()

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return str(self._device)


class MatEmbedder(HuggingMoleculesEmbedder):
    FEAT_CLS = MatFeaturizer
    MODEL_CLS = MatModel
    OUTPUT_DIM = 1024
    NEEDS_CONFORMERS = True

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.__strip_model()

    def __strip_model(self):
        self._model.generator.proj = torch.nn.Identity()


class RMatEmbedder(HuggingMoleculesEmbedder):
    FEAT_CLS = RMatFeaturizer
    MODEL_CLS = RMatModel
    OUTPUT_DIM = 3072
    NEEDS_CONFORMERS = True

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._batch_size = 2
        self.__strip_model()

    def __strip_model(self):
        self._model.generator.proj = torch.nn.Identity()


class GroverEmbedder(HuggingMoleculesEmbedder):
    FEAT_CLS = GroverFeaturizer
    MODEL_CLS = GroverModel

    def __init__(self, model_path: str):
        super().__init__(model_path)
        # atom-view and bond-view readouts are concatenated
        self.OUTPUT_DIM = 2 * self._model.config.d_model
        self.__strip_model()

    def process_batch(self, smiles):
        batch = self._featurizer(smiles).to(self._device)
        with torch.no_grad():
            atom_ffn_output, bond_ffn_output = self._model(batch)
        return torch.cat([atom_ffn_output, bond_ffn_output], 1)

    def __strip_model(self):
        self._model.mol_atom_from_atom_ffn = torch.nn.Identity()
        self._model.mol_atom_from_bond_ffn = torch.nn.Identity()


def get_embedder(name, **_kwargs) -> HuggingMoleculesEmbedder:
    if 'rmat' in name.lower():
        return RMatEmbedder(name)
    elif 'mat' in name.lower():
        return MatEmbedder(name)
    elif 'grover' in name.lower():
        return GroverEmbedder(name)
    raise ValueError(f'Unknown model name: {name}')
