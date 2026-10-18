import logging as log

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles

from chemeleon_fingerprint import CheMeleonFingerprint
from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

RDLogger.DisableLog("rdApp.*")


class ChemeleonEmbedder(SmilesEmbedder):
    def __init__(self, model_name: str = "chemeleon", batch_size: int = 128):
        self._model_name = model_name
        self._device = get_device()
        self._batch_size = batch_size
        self._fingerprint = CheMeleonFingerprint(device=self._device)
        self._output_dim = self._fingerprint.model.message_passing.output_dim

    def _embed_valid(self, smiles_batch):
        return self._fingerprint(smiles_batch)

    def _safe_batch(self, smiles_batch):
        valid_indices = []
        valid_smiles = []
        for i, smile in enumerate(smiles_batch):
            if MolFromSmiles(smile) is not None:
                valid_indices.append(i)
                valid_smiles.append(smile)
            else:
                log.error(f"Failed to parse SMILES '{smile}'")

        out = np.full((len(smiles_batch), self._output_dim), np.nan, dtype=np.float32)
        if not valid_smiles:
            return out

        try:
            embeds = self._embed_valid(valid_smiles)
            out[np.asarray(valid_indices)] = embeds
            return out
        except Exception as e:
            log.warning(f"Batch failed, processing each smile individually: {e}")
            for idx, smile in zip(valid_indices, valid_smiles):
                try:
                    out[idx] = self._embed_valid([smile])[0]
                except Exception as err:
                    log.error(f"Failed to process smile '{smile}': {err}")
            return out

    def forward(self, smiles):
        if hasattr(smiles, "tolist"):
            smiles = smiles.tolist()
        else:
            smiles = list(smiles)

        parts = [self._safe_batch(b) for b in batch(smiles, self._batch_size)]
        return np.concatenate(parts, axis=0)

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return str(self._device)


def get_embedder(name, **_kwargs):
    if "chemeleon" in name.lower():
        return ChemeleonEmbedder(model_name=name)
    raise ValueError(f"Unknown embedder {name}")
