import torch
import logging as log

from os.path import join
from hydra.utils import get_original_cwd
from src.common.types import SmilesEmbedder
from src.common.utils import batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from graphmvp_repo.src_classification.models import GNN
from graphmvp_repo.src_regression.models_complete_feature import GNNComplete
from graphmvp_repo.src_classification.config import args
from graphmvp_repo.src_classification.datasets.molecule_datasets import mol_to_graph_data_obj_simple
# GraphMVP_* checkpoints are OGB-style ("complete feature"): 9 atom / 3 bond
# features, matching GNNComplete. The classification featurizer above emits
# only 2 atom features and pairs with GNN.
from graphmvp_repo.src_regression.datasets_complete_feature.molecule_datasets import (
    mol_to_graph_data_obj_simple as mol_to_graph_data_obj_complete,
)
from rdkit import Chem
from torch_geometric.data import Batch


class GeneralGNNWrapper(SmilesEmbedder):
    def __init__(self, model_name, pool_op, task):
        self.model = model_name
        self._pool_name = pool_op
        self._pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool_op]
        self._device = 'cpu'
        model_params = {
            'num_layer': 5,  # Standard GraphMVP architecture
            'emb_dim': 300,  # Matches embedding dimension from error
            'JK': "last",    # Standard jumping knowledge
            'drop_ratio': 0.0,
            'gnn_type': "gin"
        }
        
        if task == 'classification':
            log.info("Using classification model")
            self._base_model = GNN(**model_params)
        else:
            self._base_model = GNNComplete(**model_params)
        self._model_path = join(get_original_cwd(), "model_wrappers/graphmvp/weights/", f"{model_name.split('-')[0]}.pth")
        self._base_model.load_state_dict(torch.load(self._model_path, map_location=torch.device('cpu')))
    
    
    def _forward_step(self, smiles):
        pyg_graphs = [
            mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smile))
            for smile in smiles
        ]
        batch = Batch.from_data_list(pyg_graphs)
        
        node_representation = self._base_model(batch)
        rs = self._pool(node_representation, batch=batch.batch)
        return rs

    def forward(self, smiles):
        emb = torch.cat([self._forward_step(graph) for graph in batch(smiles, n=256)], dim=0)
        return emb.detach().cpu().numpy()

    @property
    def name(self):
        return f"GNN-{self.model}"
    
    @property
    def device_used(self):
        return self._device  # Returns 'cpu' as the model is loaded on CPU


class GraphMVPWrapper(SmilesEmbedder):
    def __init__(self, model, model_name, pool_op, task):
        self.model = model
        self._pool_name = pool_op
        self._pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool_op]
        self._model_name = model_name
        self._device = 'cpu'
        model_params = {
            'num_layer': 5,  # Standard GraphMVP architecture
            'emb_dim': 300,  # Matches embedding dimension from error
            'JK': "last",    # Standard jumping knowledge
            'drop_ratio': 0.0,
            'gnn_type': "gin"
        }
        
        # Always GNNComplete: the shipped GraphMVP_* checkpoints are OGB-style
        # (99 keys, atom_encoder/bond_encoder). Building GNN here for the
        # classification task fails to load them outright -- GNN expects the
        # 57-key x_embedding/edge_embedding layout of the GNN-* baselines.
        log.info("Using complete-feature model for GraphMVP checkpoint")
        self._base_model = GNNComplete(**model_params)
        self._model_path = join(get_original_cwd(), "model_wrappers/graphmvp/weights/", model)
        self._base_model.load_state_dict(torch.load(self._model_path, map_location=torch.device('cpu')))

    def _forward_step(self, smiles):
        pyg_graphs = [
            mol_to_graph_data_obj_complete(Chem.MolFromSmiles(smile))
            for smile in smiles
        ]
        batch = Batch.from_data_list(pyg_graphs)
        
        node_representation = self._base_model(batch)
        rs = self._pool(node_representation, batch=batch.batch)
        return rs

    def forward(self, smiles):
        emb = torch.cat([self._forward_step(graph) for graph in batch(smiles, n=256)], dim=0)
        return emb.detach().cpu().numpy()

    @property
    def name(self):
        return self._model_name
    
    @property
    def device_used(self):
        return self._device  # Returns 'cpu' as the model is loaded on CPU


def get_embedder(name, task, **_kwargs):
    model_name = name.split('-')[0]
    pool_op = name.split('-')[1]

    if "graphmvp" in name.lower():
        model = f"{task}/{model_name}/pretraining_model.pth"
        return GraphMVPWrapper(model=model, model_name=name, pool_op=pool_op, task=task)
    return GeneralGNNWrapper(model_name=name, pool_op=pool_op, task=task)

    raise ValueError(f"Embedder {name} not found")
