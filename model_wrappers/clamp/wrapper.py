import os
from pathlib import Path

import torch
from hydra.utils import get_original_cwd

from src.common.types import SmilesEmbedder
from src.common.utils import get_device

#   CLAMP          - the weights released by the paper's authors (downloaded on
#                    first use), pretrained on PubChem18 with the authors'
#                    MoleculeNet test-set-only overlap filter.
#   CLAMP_refitted - our retraining on PubChem18 without any molecule of the
#                    benchmark datasets (train/valid/test), the CLAMP of the
#                    paper; weights from the Hugging Face Hub.
ORIGINAL = "CLAMP"
REFITTED = "CLAMP_refitted"
VARIANTS = (ORIGINAL, REFITTED)

# path_dir in clamp_repo defaults to a CWD-relative './data/models/...'. Hydra
# chdir's into a per-job output directory, so that would re-download the 667MB
# checkpoint for every single job; anchor it to the repo root.
_ORIGINAL_WEIGHTS = "model_wrappers/clamp/weights/clamp_clip"

# Pinned to the commit whose checkpoint.pt / hp.json produced the paper's results
# (checkpoint.pt sha256 9e245d43...e4517). snapshot_download caches the files in
# the Hugging Face cache, so every Hydra job after the first reuses them.
REFITTED_REPO = "scikit-fingerprints/benchmarking_models_clamp_refitted"
REFITTED_REVISION = "f2726330a3c7847cc58811655fbae705e23b7b6d"


def _repo_root() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:
        # get_original_cwd() only works inside a hydra run; fall back to the
        # repo root inferred from this file's location.
        return Path(__file__).resolve().parents[2]


def _build_model(variant: str, device):
    if variant == ORIGINAL:
        path_dir = _repo_root() / _ORIGINAL_WEIGHTS
        # clamp_repo's download_weights_if_not_present() only checks whether the
        # directory exists, not the files in it, so a partial/failed download is
        # silently accepted and later surfaces as an opaque torch.load error.
        if path_dir.is_dir():
            missing = [f for f in ("checkpoint.pt", "hp.json") if not (path_dir / f).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{path_dir} exists but is missing {missing}. It is an incomplete "
                    f"download; delete the directory to re-download it."
                )

        from clamp_repo.clamp.models.pretrained import PretrainedCLAMP

        # device must be passed to the constructor, not only via .to(): the
        # class keeps its own self.device attribute which nn.Module.to() does
        # not touch, and prepro_smiles() puts the input tensor on self.device.
        model = PretrainedCLAMP(path_dir=str(path_dir) + os.sep, device=str(device))
    else:
        from huggingface_hub import snapshot_download

        path_dir = Path(snapshot_download(REFITTED_REPO, revision=REFITTED_REVISION,
                                          allow_patterns=["checkpoint.pt", "hp.json"]))
        model = _build_from_checkpoint(path_dir, device)

    model.to(device)
    model.eval()
    return model


def _build_from_checkpoint(path_dir: Path, device):
    """Load our own checkpoint (checkpoint.pt + hp.json in path_dir).

    PretrainedCLAMP's (MLPLayerNorm, Pretrained) MRO trick cannot be reused
    here: `Scaled` defines its own __init__, so super().__init__(path_dir,
    device) would be misrouted into it. Build the model with its real
    signature instead and borrow Pretrained's SMILES helpers.
    """
    import json
    import types

    from clamp_repo.clamp.models.models import MLPLayerNorm
    from clamp_repo.clamp.models.pretrained import Pretrained
    from clamp_repo.clamp.models.scaled import ScaledMLPLayerNorm

    with open(path_dir / 'hp.json') as f:
        hp = json.load(f)
    classes = {'ScaledMLPLayerNorm': ScaledMLPLayerNorm, 'MLPLayerNorm': MLPLayerNorm}
    if hp['model'] not in classes:
        raise ValueError(f"{path_dir}/hp.json: unsupported model {hp['model']!r}")
    positional = {'compound_features_size', 'assay_features_size', 'embedding_size'}
    model = classes[hp['model']](
        compound_features_size=hp['compound_features_size'],
        assay_features_size=hp['assay_features_size'],
        embedding_size=hp['embedding_size'],
        **{k: v for k, v in hp.items() if k not in positional},
    )
    model.hparams = hp
    model.device = str(device)

    cp = torch.load(path_dir / 'checkpoint.pt', map_location=str(device), weights_only=False)
    # strict=True on purpose: Pretrained uses strict=False, which silently
    # tolerates a half-initialised model on any architecture mismatch.
    model.load_state_dict(cp['model_state_dict'], strict=True)

    # encode_smiles/prepro_smiles only need self.hparams, self.device and
    # self.compound_encoder, all of which are set above.
    model.prepro_smiles = types.MethodType(Pretrained.prepro_smiles, model)
    model.encode_smiles = types.MethodType(Pretrained.encode_smiles, model)
    return model


class CLAMPEmbedder(SmilesEmbedder):
    def __init__(self, variant: str = ORIGINAL):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown CLAMP variant {variant!r}; expected one of {list(VARIANTS)}")
        self._variant = variant
        self._device = get_device()
        self._model = _build_model(variant, self._device)

    def forward(self, smiles):
        import numpy as np

        smiles = list(smiles)
        out = []
        bs = 2048
        for i in range(0, len(smiles), bs):
            emb = self._model.encode_smiles(smiles[i:i + bs]).detach().cpu().numpy()
            out.append(emb)
        return np.vstack(out)

    @property
    def name(self):
        return self._variant

    @property
    def device_used(self):
        return str(self._device)


def get_embedder(name, **_kwargs):
    variant = {v.lower(): v for v in VARIANTS}.get(str(name).lower())
    if variant is None:
        raise ValueError(f"Unknown embedder {name!r}; expected one of {list(VARIANTS)}")
    return CLAMPEmbedder(variant)
