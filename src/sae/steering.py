"""Request-scoped additive steering using one pretrained SAE decoder feature."""
from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Literal

from src.sae.checkpoint import MODULE

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


class FeatureSteering:
    """Attach once per generation; strength multiplies the raw decoder row."""

    def __init__(self, model: Module, weights: dict[str, Tensor], feature_id: int,
                 strength: float, stage: Literal["prefill", "decode", "both"] = "prefill",
                 positions: Literal["last", "all"] = "last") -> None:
        import torch

        if not isfinite(strength) or not 0 <= feature_id < weights["W_dec"].shape[0]:
            raise ValueError("Expected a finite strength and a valid feature ID")
        if stage not in ("prefill", "decode", "both") or positions not in ("last", "all"):
            raise ValueError("Unsupported steering stage or token selection")
        self.direction = weights["W_dec"][feature_id].detach().clone()
        if not torch.isfinite(self.direction).all() or self.direction.norm() == 0:
            raise ValueError("Invalid SAE decoder direction")
        self.strength = strength
        self.stage = stage
        self.positions = positions
        self.visits = 0
        self.patch_count = 0
        self.handle = model.get_submodule(MODULE).register_forward_hook(self.patch)

    def patch(self, module: Module, arguments: tuple[object, ...],
              output: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...] | None:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        prefill = self.visits == 0
        self.visits += 1
        if hidden.ndim != 3 or hidden.shape[-1] != self.direction.numel():
            raise ValueError("Expected batch-by-token-by-hidden decoder states")
        if not prefill and hidden.shape[1] != 1:
            raise ValueError("Decode steering requires cached single-token steps")
        selected = "prefill" if prefill else "decode"
        if self.strength == 0 or self.stage not in (selected, "both"):
            return None
        direction = self.direction.to(device=hidden.device, dtype=torch.float32)
        adjusted = hidden.clone()
        if self.positions == "all":
            adjusted = (hidden.float() + self.strength * direction).to(hidden.dtype)
        else:
            adjusted[:, -1] = (hidden[:, -1].float() + self.strength * direction).to(hidden.dtype)
        self.patch_count += 1
        return (adjusted, *output[1:]) if isinstance(output, tuple) else adjusted

    def close(self) -> None:
        self.handle.remove()
