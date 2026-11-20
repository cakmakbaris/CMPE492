"""Explicit loading of the pinned Qwen3.5-4B block-15 SAE."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils import digest

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parents[2] / "models/sae"
REVISION = "3a67187f0082a25634aa959092bb834957aeda02"
SAE_ID = "qwen-3.5-4b/btk-mat-layer-15-k-100"
CHECKPOINT = ROOT / REVISION / SAE_ID
MODULE = "model.layers.15"
WEIGHT_SHA256 = "84a8dabb8e6c44b191d2d69c5504848f686dff92c187c9a56ece765e96107fbb"


def load_weights(path: Path = CHECKPOINT, device: str = "cpu") -> dict[str, Tensor]:
    """Load existing SAE weights only when called; verify identity and layout."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((path / "cfg.json").read_text(encoding="utf-8"))
    expected_config = {
        "architecture": "jumprelu", "d_in": 2560, "d_sae": 65536,
        "apply_b_dec_to_input": True, "normalize_activations": "none",
        "reshape_activations": "none",
    }
    if any(cfg.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Unsupported SAE configuration")
    metadata = cfg.get("metadata", {})
    if metadata.get("model_name") != "Qwen/Qwen3.5-4B" or metadata.get("hook_name") != MODULE:
        raise ValueError("SAE model or hook does not match Qwen3.5-4B block 15")
    weight_file = path / "sae_weights.safetensors"
    if digest(weight_file) != WEIGHT_SHA256:
        raise ValueError("SAE weight checksum mismatch")
    weights = load_file(str(weight_file), device=device)
    expected_shapes = {
        "W_enc": (2560, 65536), "W_dec": (65536, 2560),
        "b_enc": (65536,), "b_dec": (2560,), "threshold": (65536,),
    }
    if set(weights) != set(expected_shapes):
        raise ValueError("Unexpected SAE tensors")
    for name, shape in expected_shapes.items():
        if tuple(weights[name].shape) != shape or weights[name].dtype != torch.float32:
            raise ValueError(f"Unexpected SAE tensor layout: {name}")
    return weights
