"""One local model, one POST /generate endpoint. Start explicitly as a module."""
from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.config import Settings
from src.model import MedGemmaModel, QwenModel
from src.model_server.activations import LastTokenActivations, diagnosis_value_token
from src.model_server.steering import PrefillSteering, SteeringBundle
from src.model_server.structured import GeneratedPresencePenalty, schema_processor
from src.model_server.diagnosis_spans import DiagnosisSpanCapture
from src.schemas import (GenerateChunk, GenerateDone, GenerateError,
                         GenerateRequest, GenerateResponse)
from src.utils import create_request_directory

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import TextIteratorStreamer

ROOT = Path(__file__).resolve().parents[2]
HIDDEN_ROOT = Path(__file__).resolve().parent / "hidden_states"
ENDPOINT_ROOT = Path(__file__).resolve().parent / "endpoint_activations"
MODEL_PATHS: dict[str, Path] = {
    "medgemma": ROOT / "models/medgemma-1.5-4b-it",
    "qwen3": ROOT / "models/qwen3-4b-instruct-2507",
    "qwen3.5": ROOT / "models/qwen3.5-4b",
}
selected_model: str = "medgemma"
diagnosis_capture = False
structured_sampling = False
runtime: MedGemmaModel | QwenModel | None = None
steering_path: Path | None = None
steering_bundle: SteeringBundle | None = None
generation_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the selected model once; all requests reuse the same instance."""
    global runtime, steering_bundle
    path = MODEL_PATHS[selected_model]
    if steering_path is not None:
        if selected_model != "qwen3.5":
            raise ValueError("D82 steering requires Qwen3.5-4B")
        steering_bundle = SteeringBundle(steering_path, path)
    print(f"Model: {selected_model}; local path: {path}", flush=True)
    if structured_sampling:
        print("Structured Qwen sampling: thinking=False temperature=1 top_p=0.95 top_k=20 min_p=0 presence_penalty=1.5 repetition_penalty=1 seed=492", flush=True)
    runtime = (QwenModel(path, Settings()) if selected_model in {"qwen3", "qwen3.5"}
               else MedGemmaModel(path, Settings(), "cuda"))
    try:
        runtime.load()
        yield
    finally:
        runtime = None
        import torch

        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def save_hidden_states(directory: Path, states: tuple[tuple[Tensor, ...], ...],
                       input_ids: list[int], generated_ids: list[int], model_path: Path) -> None:
    """Save native generation steps and their token positions, without rerunning inference."""
    import torch

    # Outer tuple = forward step, inner tuple = embedding output then layer outputs.
    cpu_states = tuple(tuple(tensor.detach().cpu() for tensor in step) for step in states)
    with (directory / "hidden_states.pt").open("xb") as handle:
        torch.save(cpu_states, handle)
    sequence = input_ids + generated_ids
    steps = []
    for index, step in enumerate(cpu_states):
        start = 0 if index == 0 else len(input_ids) + index - 1
        length = step[0].shape[1]
        steps.append({"step": index, "input_position_start": start,
                      "input_token_ids": sequence[start:start + length],
                      "predicted_token_id": generated_ids[index],
                      "shapes": [list(tensor.shape) for tensor in step],
                      "dtypes": [str(tensor.dtype) for tensor in step]})
    metadata = {
        "request_id": directory.name,
        "input_token_ids": input_ids, "generated_token_ids": generated_ids,
        "layout": "hidden_states[forward_step][state_index]: batch, tokens, hidden_size",
        "state_indices": "0 is embedding output; subsequent indices are decoder layer outputs",
        "token_alignment": "Step 0 processes the prompt. Later steps process the preceding generated token. "
                           "The last generated token has no hidden state unless another forward step occurs.",
        "model": model_path.name, "model_path": str(model_path.resolve()),
        "quantization": "NF4 / BF16 compute",
        "steps": steps,
    }
    with (directory / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def generate_response(request: GenerateRequest, streamer: TextIteratorStreamer | None = None,
                      stop: Event | None = None) -> GenerateResponse:
    # A sync endpoint runs in FastAPI's threadpool. The lock serializes GPU calls.
    with generation_lock:
        if runtime is None or runtime.model is None or runtime.tokenizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        token_log_probabilities: list[float] = []
        selected_ids: list[int] = []

        class RecordTokenProbability(LogitsProcessor):
            def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
                # Greedy generation: record a scalar, never retain vocabulary-size tensors.
                values = scores[0].float()
                token = int(values.argmax().item())
                selected_ids.append(token)
                token_log_probabilities.append(float((values[token] - torch.logsumexp(values, dim=-1)).item()))
                return scores

        class StopOnDisconnect(StoppingCriteria):
            def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs: object) -> bool:
                return stop is not None and stop.is_set()

        if stop is not None and stop.is_set():
            raise RuntimeError("Streaming client disconnected before generation")

        tokenizer = runtime.tokenizer
        model = runtime.model
        sample = structured_sampling and request.response_schema is not None
        if sample and request.capture_activations:
            raise HTTPException(status_code=422, detail="Sampled structured output does not support the greedy activation probability recorder")
        if request.response_schema is not None and not isinstance(runtime, QwenModel):
            raise HTTPException(status_code=422, detail="XGrammar response_schema requires the non-thinking Qwen server")
        if request.capture_diagnosis_tokens and not isinstance(runtime, QwenModel):
            raise HTTPException(status_code=422, detail="Diagnosis-span capture requires Qwen")
        processors = LogitsProcessorList()
        if request.response_schema is not None:
            processors.append(schema_processor(tokenizer, model.config.vocab_size, request.response_schema))
        if request.capture_activations:
            # Record probabilities after grammar masking so the greedy selected IDs stay aligned.
            processors.append(RecordTokenProbability())
        if request.steering_alpha is not None and steering_bundle is None:
            raise HTTPException(status_code=422, detail="Restart with an explicit --steering-bundle")
        if request.steering_alpha is not None and (request.capture_activations or request.output_hidden_states):
            raise HTTPException(status_code=422, detail="D82 audit is separate from ordinary activation captures")
        inputs = runtime.prepare_inputs(request.conversation())
        input_length = inputs["input_ids"].shape[-1]
        if sample:
            processors.insert(0, GeneratedPresencePenalty(input_length))
        if input_length > runtime.settings.max_input_tokens:
            raise HTTPException(status_code=422, detail=f"Input exceeds the {runtime.settings.max_input_tokens}-token limit; no truncation applied")
        # Retaining every layer/step costs extra VRAM. These are initial engineering limits.
        if request.output_hidden_states and (input_length > 1024 or request.max_new_tokens > 256):
            raise HTTPException(status_code=422, detail="Hidden states require at most 1024 input and 256 new tokens")
        input_ids = inputs["input_ids"][0].tolist()
        inputs = inputs.to("cuda:0")  # Accelerate hooks handle CPU modules, never use model.device=meta.
        started = perf_counter()
        capture = LastTokenActivations(model, input_length, diagnosis_capture) if request.capture_activations else None
        spans = DiagnosisSpanCapture(model, input_length) if request.capture_diagnosis_tokens else None
        steering = (PrefillSteering(model, steering_bundle, input_length, request.steering_alpha)
                    if request.steering_alpha is not None and steering_bundle is not None else None)
        try:
            with torch.inference_mode():
                if sample:
                    torch.manual_seed(492)
                    torch.cuda.manual_seed_all(492)
                output = model.generate(**inputs, max_new_tokens=request.max_new_tokens,
                                        do_sample=sample, use_cache=True, return_dict_in_generate=True,
                                        **({"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                                            "min_p": 0.0, "repetition_penalty": 1.0} if sample else {}),
                                        output_hidden_states=request.output_hidden_states,
                                        logits_processor=processors,
                                        **({"eos_token_id": tokenizer.eos_token_id} if request.response_schema is not None else {}),
                                        streamer=streamer, stopping_criteria=StoppingCriteriaList([StopOnDisconnect()]))
                if capture is not None and not (stop is not None and stop.is_set()):
                    generated = output.sequences[0][input_length:].tolist()
                    if not generated or generated != selected_ids:
                        raise ValueError("Generated tokens do not match probability records")
                    cache = output.past_key_values
                    if cache is None or cache.get_seq_length() != output.sequences.shape[-1] - 1:
                        raise ValueError("Generation cache is not aligned before the final token")
                    capture.phase = "output"
                    mask = torch.cat([inputs["attention_mask"], torch.ones(
                        (1, len(generated)), dtype=inputs["attention_mask"].dtype, device=inputs["input_ids"].device)], dim=-1)
                    final_inputs = model.prepare_inputs_for_generation(
                        output.sequences, next_sequence_length=1, past_key_values=cache,
                        attention_mask=mask, use_cache=True, logits_to_keep=1)
                    if final_inputs["input_ids"].shape[-1] != 1:
                        raise ValueError("Expected exactly one final-token forward")
                    model(**final_inputs)
                    if diagnosis_capture:
                        capture.diagnosis_token = diagnosis_value_token(tokenizer, generated)
                        if capture.diagnosis_token == 0:
                            capture.states["pre_current_diagnosis"] = capture.states["input"]
                        elif capture.diagnosis_token is not None:
                            output.past_key_values = None
                            del final_inputs, cache
                            prefix_length = input_length + capture.diagnosis_token
                            capture.phase = "pre_current_diagnosis"
                            capture.states[capture.phase] = {}
                            model(input_ids=output.sequences[:, :prefix_length],
                                  attention_mask=mask[:, :prefix_length], use_cache=False, logits_to_keep=1)
        finally:
            if spans is not None:
                spans.close()
            if steering is not None:
                steering.close()
            if capture is not None:
                capture.close()
        if stop is not None and stop.is_set():
            raise RuntimeError("Streaming client disconnected during generation")
        if steering is not None and steering.audit is None:
            raise RuntimeError("Steering hook did not observe the prefill")
        torch.cuda.synchronize(0)
        elapsed = perf_counter() - started
        generated_ids = output.sequences[0][input_length:].tolist()
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        request_id = uuid4().hex
        hidden_path = None
        activations_path = None
        if spans is not None:
            directory = create_request_directory(ROOT / "results/diagnosis_span_captures")
            spans.save(directory, tokenizer, input_ids, generated_ids)
            activations_path = str(directory)
        if capture is not None:
            directory = create_request_directory(ENDPOINT_ROOT)
            request_id = directory.name
            eos = tokenizer.eos_token_id if request.response_schema is not None else model.generation_config.eos_token_id
            eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
            capture.save(directory, runtime.path, input_ids, generated_ids, request.max_new_tokens,
                         eos_ids, token_log_probabilities,
                         [{"role": m.role, "content": m.content} for m in request.conversation()])
            activations_path = str(directory)
        if request.output_hidden_states:
            if not output.hidden_states:
                raise HTTPException(status_code=500, detail="Model did not return hidden states")
            directory = create_request_directory(HIDDEN_ROOT)
            request_id = directory.name
            save_hidden_states(directory, output.hidden_states, input_ids, generated_ids,
                               runtime.path)
            hidden_path = str(directory)
        if not text.strip():
            raise HTTPException(status_code=502, detail=f"Empty model response; request_id={request_id}")
        return GenerateResponse(request_id=request_id, text=text, generated_tokens=len(generated_ids),
                                generation_seconds=elapsed, hidden_states_path=hidden_path,
                                activations_path=activations_path,
                                steering=steering.audit if steering is not None else None)


def stream_response(request: GenerateRequest) -> Iterator[str]:
    """Forward decoded chunks, then a final result or error, on the same endpoint."""
    from transformers import TextIteratorStreamer

    if runtime is None or runtime.tokenizer is None:
        yield GenerateError(error_type="ModelNotLoaded").model_dump_json() + "\n"
        return
    streamer = TextIteratorStreamer(runtime.tokenizer, skip_prompt=True,
                                    skip_special_tokens=True, timeout=0.5)
    outcome: Queue[GenerateDone | GenerateError] = Queue()
    stop = Event()

    def worker() -> None:
        try:
            outcome.put(GenerateDone(response=generate_response(request, streamer, stop)))
        except Exception as exc:
            outcome.put(GenerateError(error_type=type(exc).__name__))
        finally:
            # Also unblock the consumer if validation/loading fails before the first token.
            streamer.end()

    thread = Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            try:
                chunk = next(streamer)
            except Empty:
                continue
            except StopIteration:
                break
            if chunk:
                yield GenerateChunk(text=chunk).model_dump_json() + "\n"
        yield outcome.get().model_dump_json() + "\n"
    finally:
        # Observed at a token boundary, not a hard cancellation of a running GPU kernel.
        stop.set()


@app.post("/generate", response_model=None)
def generate(request: GenerateRequest) -> GenerateResponse | StreamingResponse:
    if request.stream:
        return StreamingResponse(stream_response(request), media_type="application/x-ndjson")
    return generate_response(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve one persistent local model on port 8000.")
    parser.add_argument("--model", choices=tuple(MODEL_PATHS), default="medgemma",
                        help="Model to load once at startup (default: medgemma)")
    parser.add_argument("--steering-bundle", type=Path, help="Explicit D82 direction bundle; ordinary requests remain unmodified")
    parser.add_argument("--diagnosis-capture", action="store_true",
                        help="Use selected blocks and a pre-current_diagnosis endpoint for activation requests")
    parser.add_argument("--structured-sampling", action="store_true",
                        help="Qwen schema requests: use D135 sampling/presence settings instead of greedy")
    arguments = parser.parse_args()
    if arguments.diagnosis_capture and arguments.model != "qwen3.5":
        parser.error("--diagnosis-capture requires Qwen3.5-4B")
    if arguments.structured_sampling and arguments.model != "qwen3.5":
        parser.error("--structured-sampling requires Qwen3.5-4B")
    selected_model = arguments.model
    steering_path = arguments.steering_bundle
    diagnosis_capture = arguments.diagnosis_capture
    structured_sampling = arguments.structured_sampling
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
