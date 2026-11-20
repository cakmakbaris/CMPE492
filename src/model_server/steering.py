"""Explicit, request-scoped block-22 last-prefill-token steering for D82."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.schemas import SteeringAudit
from src.utils import digest

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from transformers import PreTrainedModel


class SteeringBundle:
    def __init__(self, path: Path, model_path: Path) -> None:
        import torch

        self.sha256 = digest(path)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["model"] != "qwen3.5-4b" or data["module"] != "model.layers.21":
            raise ValueError("Steering bundle must target Qwen3.5-4B block 22")
        for name in ("config.json", "chat_template.jinja"):
            if digest(model_path / name) != data["hashes"][name]:
                raise ValueError(f"Steering bundle model mismatch: {name}")
        self.direction = data["direction"].float()
        self.sigma = float(data["sigma"])
        if (self.direction.shape != (2560,) or not torch.isfinite(self.direction).all()
                or abs(float(self.direction.norm()) - 1) > 1e-5
                or not 0 < self.sigma < float("inf")):
            raise ValueError("Invalid steering direction or scale")


class PrefillSteering:
    def __init__(self, model: PreTrainedModel, bundle: SteeringBundle,
                 input_length: int, alpha: float) -> None:
        self.bundle = bundle
        self.input_length = input_length
        self.alpha = alpha
        self.audit: SteeringAudit | None = None
        self.handle = model.get_submodule("model.layers.21").register_forward_hook(self.patch)

    def patch(self, module: Module, inputs: tuple[object, ...],
              output: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...] | None:
        if self.audit is not None:
            return None
        state = output[0] if isinstance(output, tuple) else output
        if state.shape != (1, self.input_length, 2560):
            raise ValueError("Steering requires a single full prefill")
        before = state[:, -1, :].detach().float().clone()
        if self.alpha > 0:
            # Add in float32, then round to the actual residual dtype (BF16).
            delta = self.bundle.direction.to(state.device) * (self.alpha * self.bundle.sigma)
            state[:, -1, :] = (before + delta).to(state.dtype)
        actual = float((state[:, -1, :].float() - before).norm().item())
        norm = float(before.norm().item())
        self.audit = SteeringAudit(alpha=self.alpha, bundle_sha256=self.bundle.sha256,
            input_length=self.input_length, prefill_visits=1, patch_count=int(self.alpha > 0),
            sigma=self.bundle.sigma, before_norm=norm, actual_delta_norm=actual,
            relative_delta_norm=actual / max(norm, 1e-12))
        return output if self.alpha > 0 else None

    def close(self) -> None:
        self.handle.remove()
