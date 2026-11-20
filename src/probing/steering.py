"""Replay archived Qwen prompts with score-gated linear steering and matched controls."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from shutil import copyfile
from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING

import joblib
import numpy as np
import torch

from src.config import Settings
from src.model import QwenModel
from src.schemas import ChatMessage, DiagnosisOutput
from src.probing.data import packet_path
from src.utils import create_request_directory, digest, extract_final_json, read, save

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen3.5-4b"
ARCHIVE = ROOT / "results/current_diagnosis_gru/expanded_regularized"
SOURCE = ARCHIVE / "8cf72a3c5b7244158f9f9942bc2bf0ad/manifest.json"
PROBE = ARCHIVE / "later_round_linear/475ea0a8d4a142fa9ece7359f3d5575b"
MODULE = "model.layers.21"
THRESHOLD = .5
GAIN = 2.
MAX_ALPHA = .5
SEED = 20261105


def current_field(raw: str) -> tuple[str | None, str]:
    try:
        value = json.loads(extract_final_json(raw)).get("current_diagnosis")
        if isinstance(value, str) and value.strip():
            return value, "parsed_json"
    except (ValueError, TypeError, AttributeError):
        pass
    fields = re.findall(r'"current_diagnosis"\s*:\s*("(?:[^"\\]|\\.)*")', raw)
    if len(fields) == 1:
        value = json.loads(fields[0])
        if value.strip():
            return value, "unambiguous_field_from_invalid_json"
    return None, "missing_or_ambiguous_current_diagnosis"


@dataclass(frozen=True)
class SteeringDose:
    threshold: float
    gain: float
    max_alpha: float


def archived_input(row: Mapping[str, object]) -> np.ndarray:
    path = Path(row["capture"]) / "activations.pt"
    if digest(path) != row["tensor_sha256"]:
        raise ValueError("Activation tensor hash changed")
    tensor = torch.load(path, weights_only=True, map_location="cpu")["input"][MODULE]
    if tensor.shape != (1, 2560) or not torch.isfinite(tensor).all():
        raise ValueError("Unexpected archived Input22 tensor")
    return tensor.float().numpy()[0]


def prepare(run: Path) -> None:
    probe_path = PROBE / "current_later_train.joblib"
    artifact = joblib.load(probe_path)
    scaler, probe = artifact["scaler"], artifact["model"]
    if probe.classes_.tolist() != [0, 1]:
        raise ValueError("Expected class 1 to indicate diagnosis mismatch")
    weight = probe.coef_[0] / scaler.scale_
    intercept = float(probe.intercept_[0] - weight @ scaler.mean_)
    direction = weight / np.linalg.norm(weight)
    if direction.shape != (2560,) or not np.isfinite(direction).all():
        raise ValueError("Invalid probe direction")
    split = read(PROBE / "split.json")
    patients = {p["patient_id"]: p["calls"] for p in read(SOURCE)["patients"]}
    train_vectors = [archived_input(row) for patient in split["train"] for row in patients[patient]
                     if row["call"] >= 2 and row["label"] is not None]
    training = np.stack(train_vectors)
    if int(scaler.n_samples_seen_) != len(training) or not np.allclose(scaler.mean_, training.mean(0, dtype=np.float64)):
        raise ValueError("Training vectors differ from the saved scaler")
    sigma = float(np.std(training @ direction))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid training projection scale")
    eligible = []
    candidate_count = 0
    for patient in sorted(split["test"]):
        for row in sorted(patients[patient], key=lambda r: r["call"]):
            if row["call"] < 2:
                continue
            candidate_count += 1
            x = archived_input(row)
            score = float(probe.predict_proba(scaler.transform(x.reshape(1, -1)))[0, 1])
            if score > THRESHOLD:
                eligible.append({k: row[k] for k in ("call", "packet", "capture", "request_id", "packet_sha256",
                                                     "metadata_sha256")} | {"patient_id": patient, "archived_score": score})
                break
    if len(eligible) < 20:
        raise ValueError(f"Only {len(eligible)} distinct eligible held-out patients")
    random.Random(SEED).shuffle(eligible)
    selected = eligible[:20]
    with (run / "probe_direction.pt").open("xb") as out:
        torch.save({"weight": torch.from_numpy(weight), "direction": torch.from_numpy(direction),
                    "intercept": intercept, "sigma": sigma}, out)
    save(run / "manifest.json", {"probe": str(probe_path), "probe_sha256": digest(probe_path),
         "source": str(SOURCE), "source_sha256": digest(SOURCE), "split_sha256": digest(PROBE / "split.json"),
         "direction_sha256": digest(run / "probe_direction.pt"), "test_patients": len(split["test"]),
         "eligible_test_patients": len(eligible), "examined_calls_through_first_alarm": candidate_count,
         "train_calls": len(training), "sigma": sigma, "seed": SEED, "threshold": THRESHOLD,
         "gain": GAIN, "max_alpha": MAX_ALPHA, "selection": "Earliest archived alarm on turn 2+, seed-shuffled patients. No outcome-based selection.",
         "arms": ["unpatched", "linear", "random"], "cases": selected})
    print(f"Prepared 20/{len(eligible)} eligible held-out patients, sigma={sigma:.6g}", flush=True)


def prepare_double_dose(run: Path, parent: Path) -> None:
    manifest = read(parent / "manifest.json")
    assessment = read(parent / "assessment.json")["pairs"]
    if read(parent / "finished.json")["completions"] != 60:
        raise ValueError("Follow-up requires the completed original pilot")
    originals = {r["request_id"]: r for r in replay_rows(parent) if r["arm"] == "unpatched"}
    selected = []
    for label, targets in ((1, (0.55, 0.70, 0.85)), (0, (0.60, 0.80))):
        pool = [(index, case) for index, case in enumerate(manifest["cases"], 1)
                if assessment[case["request_id"]]["unpatched"]["mismatch"] == label
                and assessment[case["request_id"]]["unpatched"]["format_valid"]]
        for target in targets:
            index, case = min(pool, key=lambda item: abs(originals[item[1]["request_id"]]["score_before"] - target))
            pool.remove((index, case))
            baseline = originals[case["request_id"]]
            selected.append({**case, "seed": manifest["seed"] + index, "baseline_label": label,
                             "baseline_prefill_sha256": baseline["prefill_tensor_sha256"]})
    direction = parent / "probe_direction.pt"
    if digest(direction) != manifest["direction_sha256"]:
        raise ValueError("Original direction changed")
    copyfile(direction, run / "probe_direction.pt")
    save(run / "manifest.json", {**manifest, "gain": 2 * manifest["gain"], "max_alpha": 2 * manifest["max_alpha"],
         "cases": selected, "arms": ["linear", "random"], "parent_run": str(parent.resolve()),
         "parent_replay_sha256": digest(parent / "replay_results.jsonl"),
         "selection": "Valid unpatched schema. Closest unused score to .55/.70/.85 among mismatches and .60/.80 among matches."})
    print(f"Prepared {len(selected)} patients, scores {[round(c['archived_score'], 4) for c in selected]}", flush=True)


def replay_rows(run: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (run / "replay_results.jsonl").read_text(encoding="utf-8").splitlines()]


def replay_call(runtime: QwenModel, messages: list[ChatMessage], metadata: dict[str, object],
                coefficients: dict[str, object], arm: str, seed: int, dose: SteeringDose,
                expected_prefill: str | None, *, module_name: str = MODULE,
                fixed_magnitude: float | None = None,
                generated_prefix: list[int] | None = None, steer_steps: int = 1) -> dict[str, object]:
    assert runtime.model is not None and runtime.tokenizer is not None
    inputs = runtime.prepare_inputs(messages)
    input_ids = inputs["input_ids"][0].tolist()
    if input_ids != metadata["input_token_ids"]:
        raise ValueError("Live and archived prompt token IDs differ")
    prefix = generated_prefix or []
    if prefix:
        inputs["input_ids"] = torch.cat((inputs["input_ids"], inputs["input_ids"].new_tensor([prefix])), dim=-1)
        mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat((mask, mask.new_ones((1, len(prefix)))), dim=-1)
        input_ids += prefix
    if len(input_ids) > runtime.settings.max_input_tokens:
        raise ValueError("Archived input exceeds the configured context bound")
    inputs = inputs.to("cuda:0")
    weight = coefficients["weight"].to("cuda:0")
    direction = coefficients["direction"].to(device="cuda:0", dtype=torch.float32)
    if arm == "random":
        direction = torch.randn(2560, generator=torch.Generator().manual_seed(seed)).to("cuda:0")
        direction /= torch.linalg.vector_norm(direction)
    observations: list[Tensor] = []
    prefill_hash = ""
    first_magnitude = 0.0

    def steer(_module: Module, _args: tuple[object, ...], output: Tensor | tuple[Tensor, ...]) -> None:
        nonlocal prefill_hash, first_magnitude
        if len(observations) >= steer_steps:
            return
        state = output[0] if isinstance(output, tuple) else output
        first_step = not observations
        if state.shape != (1, len(input_ids) if first_step else 1, 2560) or state.dtype != torch.bfloat16:
            raise ValueError("Unexpected generation shape or precision")
        before = state[0, -1].float()
        if first_step:
            prefill_hash = sha256(before.cpu().numpy().tobytes()).hexdigest()
            if expected_prefill is not None and prefill_hash != expected_prefill:
                raise ValueError("Live prefill differs from the paired baseline")
        logit = (weight @ before.double()) + coefficients["intercept"]
        score = logit.sigmoid()
        magnitude = fixed_magnitude
        if magnitude is None:
            alpha = min(dose.max_alpha, dose.gain * max(0., float(score.item()) - dose.threshold))
            magnitude = alpha * coefficients["sigma"]
        if arm == "unpatched":
            magnitude = 0.
        if magnitude:
            state[0, -1] = (before - magnitude * direction).to(state.dtype)
        after = state[0, -1].float()
        post_logit = (weight @ after.double()) + coefficients["intercept"]
        actual = torch.linalg.vector_norm(after - before)
        observations.append(torch.stack((score, post_logit.sigmoid(), actual,
                                         actual / torch.linalg.vector_norm(before))))
        if first_step:
            first_magnitude = magnitude

    handle = runtime.model.get_submodule(module_name).register_forward_hook(steer)
    start = perf_counter()
    generation = metadata["generation"]
    if generation["do_sample"] or not generation["use_cache"]:
        handle.remove()
        raise ValueError("Archived call was not greedy cached decoding")
    try:
        with torch.inference_mode():
            sequence = runtime.model.generate(**inputs, max_new_tokens=generation["max_new_tokens"] - len(prefix),
                                              do_sample=False, use_cache=True)[0]
    finally:
        handle.remove()
    if not observations:
        raise ValueError("No prefill was observed")
    generated_ids = prefix + sequence[len(input_ids):].tolist()
    if len(observations) != min(steer_steps, len(generated_ids) - len(prefix)):
        raise ValueError("Patch step count differs from generated tokens")
    trace = torch.stack(observations).cpu().tolist()
    before, after, actual, relative = trace[0]
    audit = {"score_before": before, "score_after": after, "nominal_delta_norm": first_magnitude,
             "actual_delta_norm": actual, "actual_relative_delta_norm": relative,
             "prefill_tensor_sha256": prefill_hash}
    if steer_steps > 1:
        audit["steering_steps"] = [
            {"generated_token": i, "score_before": row[0], "score_after": row[1],
             "actual_delta_norm": row[2], "actual_relative_delta_norm": row[3]}
            for i, row in enumerate(trace, len(prefix) + 1)
        ]
    return {"arm": arm, "text": runtime.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "token_count": len(generated_ids), "generated_token_ids": generated_ids,
            "elapsed_seconds": perf_counter() - start, "input_token_count": len(input_ids), **audit}


def run_pilot(run: Path) -> None:
    manifest = read(run / "manifest.json")
    dose = SteeringDose(manifest["threshold"], manifest["gain"], manifest["max_alpha"])
    direction = run / "probe_direction.pt"
    if digest(direction) != manifest["direction_sha256"]:
        raise ValueError("Prepared coefficients changed")
    coefficients = torch.load(direction, map_location="cpu", weights_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this process")
    free, total = torch.cuda.mem_get_info(0)
    if free < 2500 * 1024**2:
        raise RuntimeError(f"Insufficient free GPU memory for one Qwen instance: {free / 1024**2:.0f}/{total / 1024**2:.0f} MiB")
    runtime = QwenModel(MODEL, Settings())
    runtime.load()
    with (run / "replay_results.jsonl").open("x", encoding="utf-8") as out:
        for index, case in enumerate(manifest["cases"], 1):
            packet_file = packet_path(SOURCE.parent, Path(case["packet"]))
            metadata_path = Path(case["capture"]) / "metadata.json"
            if digest(packet_file) != case["packet_sha256"] or digest(metadata_path) != case["metadata_sha256"]:
                raise ValueError("Prompt packet or capture metadata changed")
            packet, metadata = read(packet_file), read(metadata_path)
            if packet["request_id"] != case["request_id"] or metadata["request_id"] != case["request_id"]:
                raise ValueError("Archive request identity mismatch")
            if metadata["model"] != MODEL.name or digest(MODEL / "config.json") != metadata["model_config_sha256"] or digest(MODEL / "chat_template.jinja") != metadata["chat_template_sha256"]:
                raise ValueError("Model or chat template differs from capture")
            messages = [ChatMessage.model_validate(m) for m in packet["input_messages"]]
            encoded = json.dumps([m.model_dump() for m in messages], ensure_ascii=False, separators=(",", ":"))
            if sha256(encoded.encode()).hexdigest() != metadata["conversation_sha256"]:
                raise ValueError("Prompt messages differ from capture")
            unpatched_hash = case.get("baseline_prefill_sha256")
            for arm in manifest["arms"]:
                seed = case.get("seed", manifest["seed"] + index)
                result = replay_call(runtime, messages, metadata, coefficients, arm, seed, dose, unpatched_hash)
                unpatched_hash = result["prefill_tensor_sha256"]
                record = {"patient_id": case["patient_id"], "request_id": case["request_id"],
                          "call": case["call"], "archived_score": case["archived_score"], **result}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{index}/{len(manifest['cases'])} {arm}: {result['token_count']} tokens, {result['elapsed_seconds']:.1f}s", flush=True)
    save(run / "finished.json", {"patients": len(manifest["cases"]),
         "completions": len(manifest["cases"]) * len(manifest["arms"]), "evaluation": "pending"})


def summarize(run: Path) -> None:
    manifest = read(run / "manifest.json")
    expected = {(case["request_id"], arm) for case in manifest["cases"] for arm in manifest["arms"]}
    if read(run / "finished.json")["completions"] != len(expected) or digest(SOURCE) != manifest["source_sha256"]:
        raise ValueError("Expected a complete run against the original annotation manifest")
    results = replay_rows(run)
    if len(results) != len(expected) or {(r["request_id"], r["arm"]) for r in results} != expected:
        raise ValueError("Expected one replay per patient and arm")
    if "parent_run" in manifest:
        parent = Path(manifest["parent_run"])
        if digest(parent / "replay_results.jsonl") != manifest["parent_replay_sha256"]:
            raise ValueError("Original control outputs changed")
        ids = {case["request_id"] for case in manifest["cases"]}
        controls = [dict(row, arm=row["arm"] if row["arm"] == "unpatched" else f"{row['arm']}_original")
                    for row in replay_rows(parent) if row["request_id"] in ids]
        results = controls + results
    archive = {row["request_id"]: row for patient in read(SOURCE)["patients"] for row in patient["calls"]}
    reviewed = []
    blinded = {}
    for row in results:
        original = archive[row["request_id"]]
        diagnosis, _ = current_field(row["text"])
        try:
            DiagnosisOutput.model_validate_json(extract_final_json(row["text"]))
            valid = True
        except ValueError:
            valid = False
        annotation = None
        archive_name = original["current_diagnosis"]
        if diagnosis is not None and archive_name is not None and diagnosis.strip().casefold() == archive_name.strip().casefold():
            annotation = original["label"]
        review_key = None
        if diagnosis and annotation is None:
            review_key = sha256((row["request_id"] + diagnosis.strip().casefold()).encode()).hexdigest()[:16]
            blinded[review_key] = {"diagnosis": diagnosis, "reference": original["reference"]}
        reviewed.append({k: row[k] for k in ("request_id", "call", "arm", "score_before", "score_after",
                                              "nominal_delta_norm", "actual_delta_norm", "token_count", "elapsed_seconds")}
                        | {"current_diagnosis": diagnosis, "format_valid": valid,
                           "archived_annotation_on_identical_name": annotation, "review_key": review_key})
    save(run / "evaluation_preparation.json", reviewed)
    save(run / "blinded_review_needed.json", dict(sorted(blinded.items())))
    print(f"Prepared assessment of {len(results)} outputs. {len(blinded)} unique diagnoses require review.", flush=True)


def assess(run: Path) -> None:
    rows = read(run / "evaluation_preparation.json")
    pending = read(run / "blinded_review_needed.json")
    labels = read(run / "blinded_labels.json") if pending else {}
    if set(labels) != set(pending) or any(v is not None and (type(v) is not int or v not in (0, 1))
                                          for v in labels.values()):
        raise ValueError("Blind review must assign 0, 1 or null to every pending diagnosis")
    paired: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        row["mismatch"] = (labels[row["review_key"]] if row["review_key"] is not None
                           else row["archived_annotation_on_identical_name"])
        paired.setdefault(row["request_id"], {})[row["arm"]] = row
    summary = {}
    for arm in next(iter(paired.values())):
        treatment = [case[arm] for case in paired.values()]
        comparisons = [(case["unpatched"]["mismatch"], case[arm]["mismatch"]) for case in paired.values()]
        summary[arm] = {
            "reference_matches": sum(row["mismatch"] == 0 for row in treatment),
            "reference_mismatches": sum(row["mismatch"] == 1 for row in treatment),
            "ambiguous_or_missing": sum(row["mismatch"] is None for row in treatment),
            "valid_schema_outputs": sum(row["format_valid"] for row in treatment),
            "fixed_vs_unpatched": comparisons.count((1, 0)),
            "regressed_vs_unpatched": comparisons.count((0, 1)),
            "ambiguous_pairs": sum(None in pair for pair in comparisons),
            "mean_probe_score_change": mean(row["score_after"] - row["score_before"] for row in treatment),
            "mean_actual_delta_norm": mean(row["actual_delta_norm"] for row in treatment),
        }
    save(run / "assessment.json", {"patients": len(paired), "arms": summary, "pairs": paired})
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "prepare-double-dose", "run", "summarize", "assess"), required=True)
    parser.add_argument("--output", type=Path, help="Existing run for replay/review; optional for preparation")
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare-double-dose" and args.parent is None:
        parser.error("--parent is required for prepare-double-dose")
    if args.output is None:
        if args.phase not in {"prepare", "prepare-double-dose"}:
            parser.error("--output is required for this phase")
        args.output = create_request_directory(ROOT / "results/linear_probe_steering")
    elif args.phase in {"prepare", "prepare-double-dose"}:
        args.output.mkdir(parents=True, exist_ok=True)
    print(f"Run {args.output.resolve()}", flush=True)
    torch.set_num_threads(1)
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "prepare-double-dose":
        prepare_double_dose(args.output, args.parent)
    elif args.phase == "run":
        run_pilot(args.output)
    elif args.phase == "summarize":
        summarize(args.output)
    else:
        assess(args.output)


if __name__ == "__main__":
    main()
