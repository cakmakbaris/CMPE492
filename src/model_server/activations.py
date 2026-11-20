"""Compact pre-answer representations; never retain full generation hidden states."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils import digest, extract_final_json

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def diagnosis_value_token(tokenizer: PreTrainedTokenizerBase, generated_ids: list[int]) -> int | None:
    """Locate the first generated token containing content of the top-level diagnosis string."""
    text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    try:
        final = extract_final_json(text)
        fields = json.loads(final, object_pairs_hook=list)
        values = [value for key, value in fields if key == "current_diagnosis"]
        if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
            return None
        decoder = json.JSONDecoder()
        cursor = 1
        for key, _ in fields:
            _, key_end = decoder.raw_decode(final, cursor + len(final[cursor:]) - len(final[cursor:].lstrip()))
            value_tail = final[key_end:].lstrip()[1:].lstrip()
            value_start = len(final) - len(value_tail)
            if key == "current_diagnosis":
                target = text[:text.rindex(final) + value_start + 2]
                break
            _, value_end = decoder.raw_decode(final, value_start)
            cursor = value_end + len(final[value_end:]) - len(final[value_end:].lstrip()) + 1
        else:
            return None
    except (ValueError, RecursionError):
        return None
    low, high = 1, len(generated_ids)
    while low < high:
        mid = (low + high) // 2
        prefix = tokenizer.decode(generated_ids[:mid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if prefix.startswith(target):
            high = mid
        else:
            low = mid + 1
    return low - 1


class LastTokenActivations:
    """All decoder outputs at exact input/output endpoints; no full sequences retained."""

    def __init__(self, model: PreTrainedModel, input_length: int, diagnosis_capture: bool = False) -> None:
        self.input_length = input_length
        self.phase = "input"
        self.states: dict[str, dict[str, Tensor]] = {"input": {}, "output": {}}
        self.diagnosis_capture = diagnosis_capture
        self.diagnosis_token: int | None = None
        self.handles: list[RemovableHandle] = []
        if model.config.model_type not in {"qwen3_5_text", "gemma3_text"}:
            raise ValueError("Endpoint capture requires text-only Qwen3.5 or MedGemma")
        if diagnosis_capture and model.config.model_type != "qwen3_5_text":
            raise ValueError("Selected diagnosis capture requires Qwen3.5")
        layer_count = len(model.get_submodule("model.layers"))
        if diagnosis_capture and layer_count != 32:
            raise ValueError("Selected diagnosis capture requires a 32-block model")
        layers = (0, 1, 15, 21, 30, 31) if diagnosis_capture else range(layer_count)
        self.layer_names = [f"model.layers.{i}" for i in layers]
        self.layer_names.append("model.norm")
        self.embedding_name = ("model.embed_tokens" if not diagnosis_capture
                               and model.config.model_type == "qwen3_5_text" else None)
        names = self.layer_names if self.embedding_name is None else [self.embedding_name, *self.layer_names]
        try:
            for name in names:
                self.handles.append(model.get_submodule(name).register_forward_hook(partial(self.record, name)))
        except Exception:
            self.close()
            raise

    def record(self, name: str, module: Module, inputs: tuple[object, ...],
               output: Tensor | tuple[Tensor, ...]) -> None:
        if self.phase == "idle" or (self.phase == "output" and name == self.embedding_name):
            return
        if name in self.states[self.phase]:
            return
        state = output[0] if isinstance(output, tuple) else output
        expected_length = self.input_length if self.phase == "input" else 1
        if self.phase == "pre_current_diagnosis" and self.diagnosis_token is not None:
            expected_length = self.input_length + self.diagnosis_token
        if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] != expected_length:
            raise ValueError("Unexpected endpoint capture shape")
        self.states[self.phase][name] = state.detach()[:, -1, :].to(device="cpu", copy=True)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, directory: Path, model_path: Path, input_ids: list[int],
             generated_ids: list[int], max_new_tokens: int, eos_ids: list[int],
             log_probabilities: list[float], conversation: list[dict[str, str]]) -> None:
        import torch
        import transformers

        expected_input = set(self.layer_names)
        if self.embedding_name is not None:
            expected_input.add(self.embedding_name)
        if (set(self.states["input"]) != expected_input
                or set(self.states["output"]) != set(self.layer_names)
                or (self.diagnosis_token is not None
                    and set(self.states.get("pre_current_diagnosis", {})) != set(self.layer_names))
                or len(log_probabilities) != len(generated_ids) or not generated_ids):
            raise ValueError("Incomplete endpoint capture; refusing to save mislabeled vectors")
        with (directory / "activations.pt").open("xb") as handle:
            torch.save(self.states, handle)
        serialized = json.dumps(conversation, ensure_ascii=False, separators=(",", ":"))
        config = model_path / "config.json"
        template = model_path / "chat_template.jinja"
        metadata = {
            "capture_version": 3, "mode": "input_output_last_token_all_blocks",
            "request_id": directory.name, "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model_path.name, "model_path": str(model_path.resolve()),
            "model_config_sha256": digest(config),
            "chat_template_sha256": digest(template),
            "conversation_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(conversation[0]["content"].encode()).hexdigest()
                if conversation and conversation[0]["role"] == "system" else None,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "quantization": "NF4 / BF16 compute", "layer_names": self.layer_names,
            "input_embedding": self.embedding_name,
            "input_token_ids": input_ids, "generated_token_ids": generated_ids,
            "input_position": len(input_ids) - 1, "input_last_token_id": input_ids[-1],
            "output_position": len(input_ids) + len(generated_ids) - 1,
            "output_last_token_id": generated_ids[-1], "output_last_is_eos": generated_ids[-1] in eos_ids,
            "eos_token_ids": eos_ids,
            "finish_reason": "eos" if generated_ids[-1] in eos_ids else
                "max_new_tokens" if len(generated_ids) >= max_new_tokens else "other_stop",
            "generation": {"do_sample": False, "use_cache": True, "max_new_tokens": max_new_tokens},
            "generated_token_log_probabilities": log_probabilities,
            "log_probability_definition": "Log softmax of generation-processed scores for the selected greedy token; not raw unprocessed model logits.",
            "output_capture_method": "One extra forward of the actual last generated token using generation's cache; no extra token is generated. EOS is included when emitted.",
            "layout": "dict[input|output][module_name] -> CPU Tensor[1, hidden_size]",
            "shapes": {phase: {name: list(t.shape) for name, t in values.items()} for phase, values in self.states.items()},
            "dtypes": {phase: {name: str(t.dtype) for name, t in values.items()} for phase, values in self.states.items()},
            "note": "Input endpoint is the final serialized template token. No token filtering. No clinical correctness labels enter the model. Patient/run/round linkage remains in the client's activations.jsonl.",
        }
        if self.diagnosis_capture:
            metadata.update(
                capture_version=4, mode="selected_blocks_input_output_and_pre_current_diagnosis",
                diagnosis_value_generated_index=self.diagnosis_token,
                pre_current_diagnosis_position=(None if self.diagnosis_token is None
                                                else len(input_ids) + self.diagnosis_token - 1),
                pre_current_diagnosis_status=("missing_or_invalid" if self.diagnosis_token is None else "captured"),
                pre_current_diagnosis_capture_method="Prefix replay up to the token preceding the first diagnosis value token.",
                layout="dict[input|output|pre_current_diagnosis][module_name] -> CPU Tensor[1, hidden_size]",
            )
        with (directory / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
