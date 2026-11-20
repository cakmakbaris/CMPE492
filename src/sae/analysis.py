"""Analyze selected SAE features from one saved diagnosis-span capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from src.sae.checkpoint import CHECKPOINT, MODULE, WEIGHT_SHA256, load_weights
from src.sae.encoding import encode
from src.utils import create_request_directory, digest

if TYPE_CHECKING:
    from torch import Tensor


def analyze_capture(capture: Path, feature_ids: list[int], stage: str,
                    checkpoint: Path = CHECKPOINT) -> dict[str, object]:
    """Return token values and zero-inclusive means without loading a language model."""
    return measure(capture, feature_ids, stage, load_weights(checkpoint))


def measure(capture: Path, feature_ids: list[int], stage: str,
            weights: dict[str, Tensor]) -> dict[str, object]:
    import torch
    from torch import Tensor

    if stage not in ("input", "output_selected"):
        raise ValueError("Stage must be input or output_selected")
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("module") != MODULE:
        raise ValueError("Capture must be from zero-based block 15")
    path = capture / "activations.pt"
    saved = cast(dict[str, Tensor], torch.load(path, map_location="cpu", weights_only=True))
    hidden = saved[stage]
    token_ids = metadata["input_token_ids"] if stage == "input" else [
        metadata["output_token_ids"][index] for index in metadata["selected_output_indices"]]
    if hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[0] != len(token_ids):
        raise ValueError("Activation rows and token IDs must align and be nonempty")
    if stage == "output_selected":
        if saved["output_token_indices"].tolist() != metadata["selected_output_indices"]:
            raise ValueError("Output token indices do not match capture metadata")
    with torch.inference_mode():
        features = torch.cat([encode(hidden[i:i + 32], weights, feature_ids)
                              for i in range(0, len(token_ids), 32)])
    return {
        "capture": str(capture.resolve()), "capture_sha256": digest(path),
        "metadata_sha256": digest(capture / "metadata.json"),
        "module": MODULE, "stage": stage, "sae_sha256": WEIGHT_SHA256,
        "feature_ids": feature_ids, "token_ids": token_ids,
        "positions": list(range(len(token_ids))) if stage == "input" else metadata["selected_output_indices"],
        "token_feature_values": features.tolist(),
        "mean_including_zeros": features.mean(dim=0).tolist(),
        "maximum": features.max(dim=0).values.tolist(),
        "active_fraction": (features > 0).float().mean(dim=0).tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Directory containing activations.pt and metadata.json")
    parser.add_argument("--features", type=int, nargs="+", required=True)
    parser.add_argument("--stage", choices=("input", "output_selected"), default="input")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args(argv)
    result = analyze_capture(args.capture, args.features, args.stage, args.checkpoint)
    output = create_request_directory(Path(__file__).resolve().parents[2] / "results/sae_analysis")
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
