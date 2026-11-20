"""JumpReLU encoding of previously captured model activations."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def encode(hidden: Tensor, weights: dict[str, Tensor],
           feature_ids: list[int] | None = None) -> Tensor:
    """Read all or selected features using the checkpoint's learned thresholds."""
    import torch

    encoder = weights["W_enc"]
    if hidden.ndim != 2 or hidden.shape[1] != encoder.shape[0]:
        raise ValueError("Expected a token-by-hidden-dimension activation matrix")
    if hidden.device != encoder.device:
        raise ValueError("Activations and SAE weights must be on the same device")
    if not torch.isfinite(hidden).all():
        raise ValueError("Nonfinite input activation")
    bias, threshold = weights["b_enc"], weights["threshold"]
    if feature_ids is not None:
        if not feature_ids or len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Select nonempty, distinct feature IDs")
        if min(feature_ids) < 0 or max(feature_ids) >= encoder.shape[1]:
            raise ValueError("Feature ID outside the SAE dictionary")
        indices = torch.tensor(feature_ids, device=encoder.device, dtype=torch.long)
        encoder = encoder.index_select(1, indices)
        bias = bias.index_select(0, indices)
        threshold = threshold.index_select(0, indices)
    pre = (hidden.float() - weights["b_dec"]) @ encoder + bias
    features = torch.relu(pre) * (pre > threshold)
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite SAE features")
    return features
