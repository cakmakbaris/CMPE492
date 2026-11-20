"""Local model loading and input preparation."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.config import Settings
from src.model_server.client import HttpModel as HttpModel  # Historical import path.
from src.schemas import ChatMessage, GenerateRequest

if TYPE_CHECKING:
    from transformers import BatchEncoding, BatchFeature, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase


def nf4_config() -> BitsAndBytesConfig:
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("NF4 loading requires a CUDA GPU with BF16 support")
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)


def load_nf4_model(path: Path) -> PreTrainedModel:
    """Load only MedGemma's text decoder; never instantiate its vision modules."""
    if not path.is_dir():
        raise ValueError("Provide an existing local model directory; downloads are disabled")
    model = load_medgemma_text(path, "cuda", nf4_config())
    if not model.is_loaded_in_4bit:
        raise RuntimeError("Model was not loaded in 4-bit mode")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("Text weights must remain resident on CUDA")
    return model


def load_medgemma_text(path: Path, device: str,
                       quantization: BitsAndBytesConfig | None = None) -> PreTrainedModel:
    """Map original local multimodal checkpoint keys into a text-only Gemma decoder."""
    import torch
    from transformers import AutoConfig, Gemma3ForCausalLM

    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    if config.model_type != "gemma3" or config.text_config.model_type != "gemma3_text":
        raise ValueError("Expected a local MedGemma 1.5 Gemma3 checkpoint")
    model, loading = Gemma3ForCausalLM.from_pretrained(
        str(path), config=config.text_config, local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float16,
        device_map={"": 0 if device == "cuda" else device},
        quantization_config=quantization, attn_implementation="sdpa",
        key_mapping={r"^language_model\.model\.": "model.",
                     r"^language_model\.lm_head\.": "lm_head.",
                     r"^model\.language_model\.": "model."},
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise RuntimeError("Text checkpoint is incomplete or incompatible; refusing random weights")
    unexpected = loading["unexpected_keys"]
    if any(not key.startswith(("model.vision_tower.", "model.multi_modal_projector.",
                               "vision_tower.", "multi_modal_projector."))
           for key in unexpected):
        raise RuntimeError("Unexpected non-visual checkpoint weights")
    model.eval()
    return model


class MedGemmaModel:
    """One shared model, explicit load boundary, direct access for future activation hooks.

    CUDA uses resident text-only NF4/BF16; CPU/MPS retain FP16 loading.
    The constructor stores settings only; load() must be called explicitly by the CLI.
    """
    def __init__(self, path: Path, settings: Settings, device: Literal["cpu", "mps", "cuda"] = "cpu") -> None:
        self.path = path
        self.settings = settings
        self.device = device
        self.model: PreTrainedModel | None = None
        self.text_tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        return self.text_tokenizer

    def prepare_inputs(self, messages: list[ChatMessage]) -> BatchFeature | BatchEncoding:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before preparing inputs")
        return self.tokenizer.apply_chat_template(
            template_messages(messages), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")

    def load(self) -> None:
        """Explicit runtime boundary: refuse remote IDs and load local files only."""
        if self.model is not None:
            return
        if not self.path.is_dir():
            raise ValueError("Provide an existing local model directory; downloads are disabled")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        if self.device == "cuda":
            model = load_nf4_model(self.path)
        else:
            model = load_medgemma_text(self.path, self.device)
        model.eval()
        self.text_tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str) -> str:
        return self.generate_messages([ChatMessage(role="user", content=prompt)])

    def generate_messages(self, messages: list[ChatMessage]) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load() explicitly before inference; mock mode never does so")
        import torch

        inputs = self.prepare_inputs(messages)
        input_length = inputs["input_ids"].shape[-1]
        if input_length > self.settings.max_input_tokens:
            raise ValueError("Input exceeds configured token budget; silent evidence truncation is disabled")
        # CPU offload can make model.device report meta. Accelerate's hooks handle
        # placement; inputs must stay on a real device, CUDA 0 for this loader.
        inputs = inputs.to("cuda:0" if self.device == "cuda" else self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.settings.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)


class QwenModel:
    """Local Qwen3.5-4B text model; load only on explicit server startup."""

    def __init__(self, path: Path, settings: Settings) -> None:
        self.path = path
        self.settings = settings
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.path.is_dir():
            raise ValueError("Provide an existing local model directory; downloads are disabled")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization = nf4_config()
        tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        device_map: dict[str, int | str] = {"": 0}
        # Transformers maps the Qwen3.5 multimodal checkpoint to its text-only
        # causal model and remaps language weights, excluding the vision tower.
        model = AutoModelForCausalLM.from_pretrained(
            str(self.path), local_files_only=True, dtype=torch.bfloat16,
            device_map=device_map, quantization_config=quantization,
        )
        if not model.is_loaded_in_4bit:
            raise RuntimeError("Model was not loaded in 4-bit mode")
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        print(f"Qwen ({self.path.name}) loaded text-only NF4/BF16 on cuda:0.", flush=True)

    def prepare_inputs(self, messages: list[ChatMessage]) -> BatchEncoding:
        if self.tokenizer is None:
            raise RuntimeError("Call load() before preparing inputs")
        GenerateRequest(messages=messages)
        chat_options: dict[str, bool] = {}
        if self.model is not None and self.model.config.model_type == "qwen3_5_text":
            # Keep direct instruction responses within the existing token budget.
            # The requested JSON reasoning fields are still generated normally.
            chat_options["enable_thinking"] = False
        # Qwen uses plain text role/content turns, not Gemma's multimodal processor.
        return self.tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            **chat_options)


def template_messages(messages: list[ChatMessage]) -> list[dict[str, str | list[dict[str, str]]]]:
    """Use the local processor's native template, including its initial system handling."""
    GenerateRequest(messages=messages)
    return [{"role": message.role, "content": [{"type": "text", "text": message.content}]}
            for message in messages]
