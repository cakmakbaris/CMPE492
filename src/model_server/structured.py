"""Compile per-request JSON constraints; importing this module loads no model or ML library."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import JsonValue

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import LogitsProcessor, PreTrainedTokenizerBase


class GeneratedPresencePenalty:
    """Penalize each token already generated in its own batch row."""

    def __init__(self, prompt_width: int) -> None:
        self.prompt_width = prompt_width

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        adjusted = scores.clone()
        for index in range(input_ids.shape[0]):
            tokens = input_ids[index, self.prompt_width:].unique()
            adjusted[index, tokens] -= 1.5
        return adjusted


def schema_processor(tokenizer: PreTrainedTokenizerBase, vocab_size: int,
                     schema: dict[str, JsonValue]) -> LogitsProcessor:
    """Return a fresh stateful matcher for one non-thinking generation."""
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor

    if tokenizer.eos_token_id is None:
        raise ValueError("Schema-constrained generation requires an explicit tokenizer EOS")
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                           stop_token_ids=[tokenizer.eos_token_id])
    compiled = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
    return LogitsProcessor(compiled)
