"""Capture one decoder block, retaining only input and diagnosis spans on disk."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class DiagnosisSpanCapture:
    def __init__(self, model: PreTrainedModel, input_length: int, *, row: int = 0,
                 left_padding: int = 0) -> None:
        self.input_length = input_length
        self.row = row
        self.left_padding = left_padding
        self.prompt: Tensor | None = None
        self.decoded: list[Tensor] = []
        self.handle = model.get_submodule("model.layers.15").register_forward_hook(self.record)

    def record(self, module: Module, inputs: tuple[object, ...], output: Tensor | tuple[Tensor, ...]) -> None:
        import torch

        state = output[0] if isinstance(output, tuple) else output
        row = state.detach()[self.row]
        if self.prompt is None:
            cpu = row[self.left_padding:].to(device="cpu", dtype=torch.bfloat16, copy=True)
            if cpu.shape[0] != self.input_length:
                raise ValueError("Expected full prefill")
            self.prompt = cpu
        else:
            if row.shape[0] != 1:
                raise ValueError("Expected cached single-token decode")
            self.decoded.append(row.to(device="cpu", dtype=torch.bfloat16, copy=True))

    def close(self) -> None:
        self.handle.remove()

    def save(self, directory: Path, tokenizer: PreTrainedTokenizerBase,
             input_ids: list[int], output_ids: list[int], *, batched: bool = False) -> None:
        import torch

        expected = len(output_ids) - 1
        if self.prompt is None or (len(self.decoded) < expected if batched else len(self.decoded) != expected):
            raise ValueError("Token/activation alignment mismatch")
        text = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        ends = [0] + [len(tokenizer.decode(output_ids[:i], skip_special_tokens=False,
                         clean_up_tokenization_spaces=False)) for i in range(1, len(output_ids) + 1)]
        spans = []
        selected: set[int] = set()
        for match in re.finditer(r'"diagnosis"\s*:\s*("(?:[^"\\]|\\.)*")', text):
            start, end = match.start(1) + 1, match.end(1) - 1
            positions = [i for i in range(len(output_ids)) if ends[i] < end and ends[i+1] > start]
            if not positions:
                raise ValueError("Empty diagnosis token span")
            before = positions[0] - 1
            selected.update(positions)
            if before >= 0:
                selected.add(before)
            spans.append({"diagnosis": json.loads(match.group(1)), "character_span": [start, end],
                          "output_token_indices": positions, "pre_diagnosis_output_index": before,
                          "negative_index_means": "last input token"})
        if not spans or (selected and max(selected) >= expected):
            raise ValueError("Diagnosis span has no processed hidden state")
        indices = sorted(selected)
        states = torch.cat(self.decoded[:expected], dim=0) if expected else torch.empty((0, self.prompt.shape[1]))
        torch.save({"input": self.prompt, "output_selected": states[indices],
                    "output_token_indices": torch.tensor(indices)}, directory / "activations.pt")
        metadata = {"module": "model.layers.15", "indexing": "zero-based block15 = sixteenth block",
            "dtype": "bfloat16", "input_token_ids": input_ids, "output_token_ids": output_ids,
            "output_tokens": tokenizer.convert_ids_to_tokens(output_ids), "diagnosis_spans": spans,
            "selected_output_indices": indices, "input_shape": list(self.prompt.shape),
            "selected_output_shape": list(states[indices].shape),
            "alignment": "Each vector is after processing its indexed token; predicts the next token. Final EOS need not have a vector.",
            "temporary_storage": "All decoded vectors copied to CPU during generation; only selected spans persisted."}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
