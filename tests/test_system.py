"""Synthetic schema, dataset-adapter, retrieval and filesystem checks."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from src.dataset import RawCase, adapt_case, load_case, load_cases
from src.retrieval import available_tests, retrieve_tests
from src.schemas import ChatMessage, DiagnosisOutput, FoundResult, GenerateRequest
from src.schemas import TestRequest as ExaminationRequest
from src.schemas import TestResult as ResultUnion
from src.utils import create_run_directory, normalize_presentation

ROOT = Path(__file__).resolve().parents[1]
RAW_CASE = {"case_summary": "Synthetic symptoms", "key_pertinent_results_dict": {"A": "exact value"}}
FINAL: dict[str, JsonValue] = {
    "action": "final_diagnosis",
    "hypotheses": [{"diagnosis": "Synthetic conclusion", "reasoning": "Illustrative evidence."}],
    "current_diagnosis": "Synthetic conclusion",
    "requests": [],
}


def request_test(name: str) -> ExaminationRequest:
    return ExaminationRequest(name=name, justification="Assess")


@pytest.mark.parametrize("source, expected", [
    ("Patient Information: ___ y/o F", "Patient Information: Female. Age: Not provided"),
    ("___ year old male; relative: ____; ?cancer", "Age: Not provided male; relative: Not provided; ?cancer"),
    ("42 y/o F; ?cancer", "42 y/o F; ?cancer"),
    ("___ y/o; sex not provided", "Age: Not provided; sex not provided"),
])
def test_presentation_redactions_are_explicit_without_resolving_uncertainty(source: str, expected: str) -> None:
    assert normalize_presentation(source) == expected


@pytest.mark.parametrize("value", [None, " ", [], {}])
def test_requested_missing_values_are_marked_and_consumed(value: JsonValue) -> None:
    with pytest.raises(ValidationError):
        FoundResult(status="found", requested_name="A", matched_key="A", recorded_value=value)
    examinations = {"A": value, "B": 0}
    output = retrieve_tests([request_test("A")], examinations, [])
    assert output.test_results[0].status == "not_provided"
    assert available_tests(examinations, output.test_results) == ["B"]


@pytest.mark.parametrize("source,requested", [
    ("Abdominal and pelvic CT.", "Abdominal and pelvic CT"),
    ("Legionella Urinary Antigen ", "Legionella Urinary Antigen"),
    ("CT cervical spine ___.", "CT cervical spine"),
    ("CT CHEST", "CT CHEST ___ .  "),
    ("URINE CULTURE", "Urine Culture"),
])
def test_trailing_format_variants_retrieve_original_value(source: str, requested: str) -> None:
    value: JsonValue = {"reading": 0, "notes": "  preserve result  "}
    examinations = {source: value, "Other": None}
    output = retrieve_tests([request_test(requested)], examinations, [])
    assert not output.rejected_requests
    result = output.test_results[0]
    assert isinstance(result, FoundResult)
    assert result.requested_name == requested
    assert result.matched_key == source
    assert result.recorded_value == value
    assert available_tests(examinations, output.test_results) == ["Other"]
    repeated = retrieve_tests([request_test(source)], examinations, output.test_results)
    assert repeated.rejected_requests[0].reason == "already_requested"


def test_trailing_variants_cannot_repeat_missing_tests_or_batch_entries() -> None:
    examinations: dict[str, JsonValue] = {"CT spine ___.": None}
    output = retrieve_tests([
        request_test("CT spine"),
        request_test("ct SPINE."),
    ], examinations, [])
    assert output.test_results[0].status == "not_provided"
    assert output.rejected_requests[0].reason == "duplicate_in_batch"
    assert available_tests(examinations, output.test_results) == []
    repeated = retrieve_tests([request_test("CT spine ___")], examinations, output.test_results)
    assert repeated.rejected_requests[0].reason == "already_requested"


@pytest.mark.parametrize("requested", ["CT  spine", "CT_spine", "CT spine,", "CT spine_"])
def test_name_normalization_does_not_guess_other_variants(requested: str) -> None:
    output = retrieve_tests([request_test(requested)], {"CT spine.": "Recorded"}, [])
    assert not output.test_results
    assert output.rejected_requests[0].reason == "not_in_catalog"


def test_colliding_normalized_keys_fail_without_selecting_a_value() -> None:
    with pytest.raises(ValueError, match="same normalized name"):
        retrieve_tests([request_test("CT")], {"CT": "First", "ct.": "Second"}, [])


def test_result_discriminator_rejects_fabricated_value() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ResultUnion).validate_python({
            "status": "not_provided",
            "requested_name": "A",
            "reason": "missing",
            "recorded_value": "fake",
        })


def test_adapter_supported_shapes_and_missing_required_fields(tmp_path: Path) -> None:
    raw = dict(RAW_CASE, stepwise_diagnostic_reasoning_timeline_list=["private"])
    path = tmp_path / "synthetic.json"
    for content, expected_id in [([raw], "patient_001"), ({"custom_id": raw}, "custom_id")]:
        path.write_text(json.dumps(content), encoding="utf-8")
        case = load_cases(path)[0]
        assert case.patient_id == expected_id
        assert case.reference.timeline == ["private"]

    path.write_text(json.dumps([{"case_summary": "Synthetic symptoms"}]), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_cases(path)

    duplicate = dict(raw, patient_id="duplicate")
    path.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(path)


def test_concurrent_run_reservation(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        reservations = [executor.submit(create_run_directory, tmp_path, "p") for _ in range(8)]
        paths = [reservation.result() for reservation in reservations]
    assert sorted(path.name for path in paths) == [f"run{i:03d}" for i in range(1, 9)]


def test_patient_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "other"
    target.mkdir()
    (tmp_path / "p").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, "p")


def test_import_and_constructor_never_import_ml_libraries() -> None:
    code = """
import sys
from pathlib import Path

from src.config import Settings
from src.model import MedGemmaModel

model = MedGemmaModel(Path("not-a-model"), Settings())
assert model.model is None
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_exact_keys_and_values_survive_adapter_and_retrieval() -> None:
    name = "  Blood count  "
    value: JsonValue = {"reading": 0, "notes": ["  keep spaces  "]}
    raw = RawCase(case_summary="Synthetic symptoms", key_pertinent_results_dict={name: value, "Zero": 0})
    case = adapt_case(raw, "p")
    output = retrieve_tests([request_test(name), request_test("Zero")], case.examinations, [])
    assert not output.rejected_requests
    assert output.test_results == [
        FoundResult(status="found", requested_name=name, matched_key=name, recorded_value=value),
        FoundResult(status="found", requested_name="Zero", matched_key="Zero", recorded_value=0),
    ]


@pytest.mark.parametrize(("content", "patient_id"), [
    pytest.param([{"patient_id": "other"}, dict(RAW_CASE, patient_id="selected")], "selected", id="list"),
    pytest.param({"other": {}, "selected": RAW_CASE}, "selected", id="mapping"),
    pytest.param([dict(RAW_CASE, patient_id=42)], "42", id="numeric-id"),
    pytest.param([RAW_CASE], "patient_001", id="fallback-id"),
])
def test_load_one_patient(tmp_path: Path, content: JsonValue, patient_id: str) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    case = load_case(path, patient_id)
    assert case.patient_id == patient_id
    assert case.examinations == {"A": "exact value"}


def test_single_patient_missing_duplicate_and_invalid_record(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.json"
    record = {"patient_id": "p"}
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(LookupError, match="not found"):
        load_case(path, "missing")
    with pytest.raises(ValueError, match="blank"):
        load_case(path, " ")
    with pytest.raises(ValidationError):
        load_case(path, "p")
    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_case(path, "p")


@pytest.mark.parametrize("changes, message", [
    ({"current_diagnosis": "Not a hypothesis"}, "exactly match"),
    ({"hypotheses": [
        {"diagnosis": "Synthetic conclusion", "reasoning": "One"},
        {"diagnosis": "SYNTHETIC CONCLUSION", "reasoning": "Two"}]}, "unique"),
    ({"action": "request_tests"}, "at least one"),
    ({"requests": [{"name": "A", "justification": "Seek evidence"}]}, "empty requests"),
    ({"hypotheses": []}, "at least 1 item"),
    ({"diagnosis": "Legacy output field"}, "Extra inputs"),
])
def test_single_diagnosis_output_rejects_inconsistent_fields(changes: dict[str, JsonValue], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DiagnosisOutput.model_validate(FINAL | changes)


@pytest.mark.parametrize("field", ["hypotheses", "current_diagnosis", "action", "requests"])
def test_final_diagnosis_requires_the_same_four_fields(field: str) -> None:
    payload = FINAL.copy()
    del payload[field]
    with pytest.raises(ValidationError, match="Field required"):
        DiagnosisOutput.model_validate(payload)


@pytest.mark.parametrize("messages", [[], [ChatMessage(role="assistant", content="x")],
    [ChatMessage(role="user", content="a"), ChatMessage(role="user", content="b")]])
def test_generate_request_rejects_invalid_role_order(messages: list[ChatMessage]) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(messages=messages)


def test_generate_request_accepts_legacy_prompt_or_conversation() -> None:
    assert GenerateRequest(prompt="hello").conversation() == [ChatMessage(role="user", content="hello")]
    with pytest.raises(ValidationError):
        GenerateRequest()
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hello", messages=[ChatMessage(role="user", content="hi")])


