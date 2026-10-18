"""MolE static molecular representations (gin_concat, R=1000).

Upstream: https://github.com/rolayoalarcon/MolE
Weights:  https://zenodo.org/records/10803099
"""

from __future__ import annotations

import logging as log
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, Data

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

RDLogger.DisableLog("rdApp.*")

PRETRAIN_NAME = "gin_concat_R1000_E8000_lambda0.0001"
# MolE gather_representation uses the concatenated GNN readout (r), not the BT projection (z).
OUTPUT_DIM = 1000  # num_layer (5) * emb_dim (200)

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
# GINet chirality embedding size is 3; clamp CHI_OTHER → unspecified
_MAX_CHIRALITY_IDX = 2


def _wrapper_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_mole_on_path() -> Path:
    repo = _wrapper_dir() / "mole_repo"
    if not repo.is_dir():
        raise FileNotFoundError(
            f"MolE sources missing at {repo}. Re-run ./embed_wrapper.sh model_wrappers/mole -r"
        )
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _smiles_to_graph(smiles: str) -> Data | None:
    """Build a MolE-style PyG graph (with explicit Hs), matching dataset_representation.py."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 0 or atomic_num not in ATOM_LIST:
            return None
        type_idx.append(ATOM_LIST.index(atomic_num))
        chi = CHIRALITY_LIST.index(atom.GetChiralTag())
        chirality_idx.append(min(chi, _MAX_CHIRALITY_IDX))

    x = torch.tensor(
        np.column_stack([type_idx, chirality_idx]), dtype=torch.long
    )

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir()),
        ]
        row += [start, end]
        col += [end, start]
        edge_feat.append(feat)
        edge_feat.append(feat)

    if not row:
        # Single-atom molecules: empty edge_index / edge_attr (MolE self-loops added in GINEConv)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolEEmbedder(SmilesEmbedder):
    def __init__(self, model_name: str = "mole", batch_size: int = 256):
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = get_device()
        self._output_dim = OUTPUT_DIM

        _ensure_mole_on_path()
        from models.ginet_concat import GINet

        ckpt_dir = _wrapper_dir() / "ckpt" / PRETRAIN_NAME / "checkpoints"
        config_path = ckpt_dir / "config.yaml"
        model_path = ckpt_dir / "model.pth"
        if not config_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                "MolE weights missing. Expected:\n"
                f"  {config_path}\n"
                f"  {model_path}\n"
                "Download from https://zenodo.org/records/10803099 :\n"
                f'  curl -o "{config_path}" '
                '"https://zenodo.org/records/10803099/files/config.yaml?download=1"\n'
                f'  curl -o "{model_path}" '
                '"https://zenodo.org/records/10803099/files/model.pth?download=1"'
            )

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_config = config["model"]

        self._model = GINet(**model_config).to(self._device)
        state = torch.load(model_path, map_location=self._device, weights_only=False)
        # Quiet load_my_state_dict's per-parameter prints
        import builtins

        _print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            self._model.load_my_state_dict(state)
        finally:
            builtins.print = _print
        self._model.eval()

    def _embed_graphs(self, graphs: list[Data]) -> np.ndarray:
        batch_obj = Batch.from_data_list(graphs).to(self._device)
        with torch.no_grad():
            h_rep, _ = self._model(batch_obj)
        return h_rep.detach().cpu().numpy().astype(np.float32)

    def _safe_batch(self, smiles_batch: list[str]) -> np.ndarray:
        out = np.full((len(smiles_batch), self._output_dim), np.nan, dtype=np.float32)
        valid_idx = []
        graphs = []
        for i, smi in enumerate(smiles_batch):
            try:
                g = _smiles_to_graph(smi)
            except Exception as e:
                log.error(f"Failed to featurize SMILES '{smi}': {e}")
                g = None
            if g is None:
                log.error(f"Failed to parse / featurize SMILES '{smi}'")
                continue
            valid_idx.append(i)
            graphs.append(g)

        if not graphs:
            return out

        try:
            embeds = self._embed_graphs(graphs)
            out[np.asarray(valid_idx)] = embeds
            return out
        except Exception as e:
            log.warning(f"Batch failed, processing each SMILES individually: {e}")
            for idx, graph in zip(valid_idx, graphs):
                try:
                    out[idx] = self._embed_graphs([graph])[0]
                except Exception as err:
                    log.error(f"Failed to embed SMILES index {idx}: {err}")
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
    if "mole" in name.lower():
        return MolEEmbedder(model_name=name)
    raise ValueError(f"Unknown embedder {name}")
