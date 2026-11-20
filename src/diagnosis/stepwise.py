"""Run fixed, shuffled stepwise groups with per-row activation capture."""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
import json
from os import _exit
from pathlib import Path
from sys import stderr, stdout
from traceback import print_exc
from time import perf_counter
from random import Random
from shutil import disk_usage
from contextlib import ExitStack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xgrammar as xgr
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.diagnosis.schemas import StepwiseDiagnosisOutput
from src.config import Settings
from src.dataset import load_cases
from src.model import QwenModel
from src.model_server.diagnosis_spans import DiagnosisSpanCapture
from src.model_server.structured import GeneratedPresencePenalty
from src.schemas import ChatMessage, JsonValue, PatientCase, is_missing
from src.utils import create_request_directory, extract_final_json, replace_json


@dataclass
class State:
    case: PatientCase
    catalog: dict[str, str]
    revealed: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    last: StepwiseDiagnosisOutput | None = None
    termination: str | None = None
    valid_rounds: int = 0


MAX_ROUNDS = 4
SEED = 492


def round_input(state: State, number: int, prompt: str) -> tuple[dict[str, str], dict[str, JsonValue], list[ChatMessage]]:
    """Project one patient's visible evidence and constrain this round's test IDs."""
    available = {key: name for key, name in state.catalog.items() if key not in state.revealed}
    schema = StepwiseDiagnosisOutput.model_json_schema()
    if available:
        schema["$defs"]["StepwiseTestRequest"]["properties"]["test_id"]["enum"] = list(available)
    else:
        schema["properties"]["action"]["enum"] = ["final_diagnosis"]
    schema["properties"]["requests"]["maxItems"] = min(3, len(available))
    messages = [ChatMessage(role="system", content=prompt), ChatMessage(
        role="user", content=json.dumps({"round": number, "patient_presentation": state.case.presentation,
            "available_tests": available, "revealed_tests": state.revealed}, ensure_ascii=False))]
    return available, schema, messages


def run_group(run: Path, states: list[State], prompt: str, settings: Settings,
              runtime: QwenModel, model: PreTrainedModel,
              tokenizer: PreTrainedTokenizerBase, compiler: xgr.GrammarCompiler) -> None:
    import torch
    from transformers import LogitsProcessorList
    from xgrammar.contrib.hf import LogitsProcessor as GrammarProcessor

    eos = tokenizer.eos_token_id
    total_started = perf_counter()
    total_tokens = 0
    total_seconds = 0.0
    try:
        for number in range(1, MAX_ROUNDS + 1):
            active = [state for state in states if state.termination is None]
            if not active:
                break
            if disk_usage(run).free < 6 * 1024**3:
                raise OSError("Less than 6 GiB free for activation capture")
            batch_started = perf_counter()
            compiled, prompts, folders, available_by_row = [], [], [], []
            for state in active:
                folder = run / state.case.patient_id / f"round_{number:02d}"
                folder.mkdir(parents=True)
                available, schema, messages = round_input(state, number, prompt)
                replace_json(folder / "messages.json", [message.model_dump() for message in messages])
                replace_json(folder / "response_schema.json", schema)
                prompts.append(runtime.prepare_inputs(messages)["input_ids"][0].tolist())
                compiled.append(compiler.compile_json_schema(json.dumps(schema)))
                folders.append(folder)
                available_by_row.append(available)
            inputs = tokenizer.pad({"input_ids": prompts}, return_tensors="pt").to("cuda:0")
            width = inputs["input_ids"].shape[-1]
            if width > settings.max_input_tokens:
                raise ValueError("Input exceeds configured token limit")

            print(f"Round {number}: batch={len(active)}, prompt tokens={[len(p) for p in prompts]}", flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            torch.manual_seed(492)
            torch.cuda.manual_seed_all(492)
            with ExitStack() as stack, torch.inference_mode():
                captures = []
                for index, prompt_ids in enumerate(prompts):
                    capture = DiagnosisSpanCapture(model, len(prompt_ids), row=index,
                                                   left_padding=width - len(prompt_ids))
                    captures.append(capture)
                    stack.callback(capture.close)
                output = model.generate(**inputs, max_new_tokens=settings.max_new_tokens, do_sample=True,
                    use_cache=True, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                    repetition_penalty=1.0, eos_token_id=eos,
                    logits_processor=LogitsProcessorList([GeneratedPresencePenalty(width), GrammarProcessor(compiled)]))
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 2**20
            rows = []
            for state, folder, available, completion, prompt_ids, capture in zip(
                    active, folders, available_by_row, output[:, width:].tolist(), prompts, captures, strict=True):
                if eos in completion:
                    completion = completion[:completion.index(eos) + 1]
                total_tokens += len(completion)
                raw = tokenizer.decode(completion, skip_special_tokens=True)
                (folder / "raw_output.txt").write_text(raw, encoding="utf-8")
                row = {"patient": state.case.patient_id, "tokens": len(completion)}
                try:
                    diagnosis = StepwiseDiagnosisOutput.model_validate_json(extract_final_json(raw))
                    if any(item.test_id not in available for item in diagnosis.requests):
                        raise ValueError("Test ID is not in the current catalog")
                    activation_dir = folder / "activations"
                    activation_dir.mkdir()
                    capture.save(activation_dir, tokenizer, prompt_ids, completion, batched=True)
                    state.last = diagnosis
                    state.valid_rounds = number
                    replace_json(folder / "output.json", diagnosis.model_dump(mode="json"))
                    row.update(action=diagnosis.action, requests=len(diagnosis.requests), valid=True)
                    if diagnosis.action == "final_diagnosis":
                        state.termination = "model_final"
                    elif number == MAX_ROUNDS:
                        state.termination = "max_rounds"
                    else:
                        for item in diagnosis.requests:
                            name = available[item.test_id]
                            state.revealed[item.test_id] = {"name": name, "result": state.case.examinations[name],
                                                           "revealed_at_round": number + 1}
                except ValueError as exc:
                    state.termination = "generation_or_validation_error"
                    replace_json(folder / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                    row.update(valid=False)
                rows.append(row)
            total_seconds += elapsed
            metrics = {"round": number, "batch": len(active), "model_seconds": elapsed,
                       "wall_seconds": perf_counter() - batch_started, "peak_allocated_mib": peak, "rows": rows}
            replace_json(run / f"metrics_round_{number:02d}.json", metrics)
            print(metrics, flush=True)
    except Exception as exc:
        replace_json(run / "batch_error.json", {"round": number, "batch": len(active),
                                        "type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        for state in states:
            patient_dir = run / state.case.patient_id
            patient_dir.mkdir(exist_ok=True)
            replace_json(patient_dir / "reference_for_evaluation.json", state.case.reference.model_dump(mode="json"))
        replace_json(run / "summary.json", {"wall_seconds": perf_counter() - total_started, "model_seconds": total_seconds,
            "total_tokens": total_tokens, "patients": [{"patient": s.case.patient_id, "valid_rounds": s.valid_rounds,
                "termination": s.termination, "first_hypothesis": s.last.hypotheses[0].diagnosis if s.last else None}
                for s in states]})


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument("--resume", type=Path, help="Existing run directory; continue from its last started group")
    parser.add_argument("--skip-current", action="store_true", help="With --resume, skip the interrupted group")
    parser.add_argument("--limit", type=int, help="First N eligible cases after shuffling")
    parser.add_argument("--seed", type=int, help="Patient shuffle seed (default 492)")
    parser.add_argument("--dataset", type=Path, help="Local DiagBench JSON file")
    arguments = parser.parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if arguments.resume and any(v is not None for v in (arguments.limit, arguments.seed, arguments.dataset)):
        parser.error("--resume uses the dataset, limit and seed saved in the run")
    if arguments.skip_current and not arguments.resume:
        parser.error("--skip-current requires --resume")

    from inspect import getclosurevars
    import torch
    import xgrammar as xgr
    from transformers.models.qwen3_5 import modeling_qwen3_5

    root = Path(__file__).resolve().parents[2]
    prompt = (root / "prompts/diagnosis/stepwise.txt").read_text(encoding="utf-8")
    protocol = json.loads((arguments.resume / "protocol.json").read_text(encoding="utf-8")) if arguments.resume else {}
    dataset = Path(protocol.get("dataset") or arguments.dataset or root / "data/DiagBench_MIMICIV.json")
    seed = protocol.get("seed", SEED if arguments.seed is None else arguments.seed)
    limit = protocol.get("limit", arguments.limit)
    cases = [case for case in load_cases(dataset) if not case.leakage_flags]
    Random(seed).shuffle(cases)
    cases = cases[:limit]
    groups = [cases[i:i + 4] for i in range(0, len(cases), 4)]
    planned = [[case.patient_id for case in group] for group in groups]
    failures: list[int] = []
    first_group = 1
    if arguments.resume is not None:
        run = arguments.resume.resolve()
        prompt = (run / "prompt.txt").read_text(encoding="utf-8")
        if json.loads((run / "protocol.json").read_text(encoding="utf-8"))["groups"] != planned:
            raise RuntimeError("Resume protocol does not match the deterministic group plan")
        if (run / "finished.json").exists():
            print(f"Run {run} is already finished", flush=True)
            return
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        first_group, failures = progress["group"], progress["failures"]
        # A group that crashed (batch_error.json) or kept stalling is skipped, not retried; its folder is kept for later.
        if (run / f"group_{first_group:03d}" / "batch_error.json").exists() or arguments.skip_current:
            failures.append(first_group)
            print(f"Skipping crashed group {first_group}; failed groups so far: {failures}", flush=True)
            first_group += 1
    else:
        run = create_request_directory(root / "results/stepwise_batch4_activations")
        replace_json(run / "protocol.json", {"seed": seed, "dataset": str(dataset.resolve()), "limit": limit, "groups": planned,
            "max_rounds": MAX_ROUNDS, "max_tests": 3, "activation_capture": "model.layers.15 BF16 input and diagnosis spans",
            "xgrammar": True, "fla": True, "split_on_oom": False})
        (run / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"Run {run} | {len(cases)} patients | {len(groups)} groups | starting at group {first_group}", flush=True)
    for kernel in (modeling_qwen3_5.torch_chunk_gated_delta_rule, modeling_qwen3_5.torch_recurrent_gated_delta_rule):
        implementation = getclosurevars(kernel).nonlocals.get("implementation")
        if not getattr(implementation, "__module__", "").startswith("fla."):
            raise RuntimeError("FLA is not active")
    settings = Settings()
    runtime = QwenModel(root / "models/qwen3.5-4b", settings)
    runtime.load()
    model, tokenizer = runtime.model, runtime.tokenizer
    assert model is not None and tokenizer is not None
    tokenizer.padding_side = "left"
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=model.config.vocab_size, stop_token_ids=[tokenizer.eos_token_id]))
    for index, group in enumerate(groups, 1):
        if index < first_group:
            continue
        folder = run / f"group_{index:03d}"
        if folder.exists():
            # Preserve the interrupted attempt; never overwrite or delete prior output.
            attempt = 1
            while (run / f"{folder.name}_incomplete_{attempt}").exists():
                attempt += 1
            folder.rename(run / f"{folder.name}_incomplete_{attempt}")
        folder.mkdir()
        states = []
        for case in group:
            names = [name for name, value in case.examinations.items() if not is_missing(value)]
            states.append(State(case, {f"T{i:03d}": name for i, name in enumerate(names, 1)}))
        replace_json(run / "progress.json", {"group": index, "groups": len(groups), "failures": failures})
        try:
            run_group(folder, states, prompt, settings, runtime, model, tokenizer, compiler)
        except (torch.OutOfMemoryError, ValueError) as exc:
            failures.append(index)
            print(f"Group {index} failed: {type(exc).__name__}", flush=True)
        if failures and failures[-1] == index:
            torch.cuda.empty_cache()
    replace_json(run / "finished.json", {"groups": len(groups), "failed_groups": failures})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A CUDA error leaves a hung process holding the GPU; exit hard so a watchdog can resume.
        print_exc()
        stdout.flush()
        stderr.flush()
        _exit(1)
