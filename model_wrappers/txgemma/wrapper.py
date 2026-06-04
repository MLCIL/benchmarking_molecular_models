from typing import Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

from src.common.types import SmilesEmbedder

class TxGemmaEmbedder(SmilesEmbedder):
    MODEL_PATH = "google/txgemma-2b-predict"

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        pooling: str = "mean",
        max_memory: Optional[dict] = None,
        **kwargs
    ):
        self._model_path = model_path
        self._name = model_path.split("/")[-1]
        self._batch_size = batch_size
        self._max_length = max_length
        self._pooling = pooling

        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=torch.float32,
            device_map="auto",
            max_memory=max_memory,
        )

        self._model.eval()

    @property
    def name(self) -> str:
        return self._name.split("/")[-1] + f"_{self._pooling}_pooling"

    def forward(self, smiles) -> np.ndarray:
        all_embeddings = []

        for start in tqdm(range(0, len(smiles), self._batch_size)):
            batch = smiles[start:start + self._batch_size]
            batch_embeddings = self._forward_batch(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _forward_batch(self, smiles_batch) -> np.ndarray:
        prompts = [f"SMILES: {s}" for s in smiles_batch]

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )

        first_device = next(self._model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        last_hidden_state = outputs.hidden_states[-1]

        if self._pooling == "mean":
            embeddings = self._mean_pool(last_hidden_state, inputs["attention_mask"])
        elif self._pooling == "last_token":
            embeddings = self._last_token_pool(last_hidden_state, inputs["attention_mask"])
        else:
            raise ValueError(f"Unknown pooling method: {self._pooling}")
        
        if not torch.isfinite(embeddings).all():
            invalid_mask = ~torch.isfinite(embeddings).all(dim=1)

            for s, is_invalid in zip(smiles_batch, invalid_mask):
                if is_invalid.item():
                    print(f"INVALID_SMILES: {s}")

        return embeddings.float().cpu().numpy()

    def _mean_pool(self, last_hidden_state, attention_mask):
        attention_mask = attention_mask.to(last_hidden_state.device)
        last_hidden_state = last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        return summed / counts

    def _last_token_pool(self, last_hidden_state, attention_mask):
        attention_mask = attention_mask.to(last_hidden_state.device)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            last_hidden_state.size(0),
            device=last_hidden_state.device,
        )
        return last_hidden_state[batch_indices, sequence_lengths]

def get_embedder(name, **_kwargs):
    if "txgemma" in name.lower():
        return TxGemmaEmbedder(**_kwargs)

    raise ValueError(f"Unknown embedder {name}")