"""Synthetic schema, dataset-adapter, retrieval and filesystem checks."""
from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from src.retrieval import available_tests, retrieve_tests
from src.dataset import RawCase, adapt_case, load_case, load_cases
from src.schemas import DiagnosisOutput, FoundResult, ChatMessage, GenerateRequest, TestRequest as ExaminationRequest, TestResult as ResultUnion
from src.utils import create_run_directory, normalize_presentation

ROOT = Path(__file__).resolve().parents[1]
FINAL = json.dumps({"action": "final_diagnosis", "hypotheses": [
    {"diagnosis": "Synthetic conclusion", "reasoning": "Illustrative evidence."}],
    "current_diagnosis": "Synthetic conclusion", "requests": []})


@pytest.mark.parametrize("source, expected", [
    ("Patient Information: ___ y/o F", "Patient Information: Female. Age: Not provided"),
    ("___ year old male; relative: ____; ?cancer", "Age: Not provided male; relative: Not provided; ?cancer"),
    ("42 y/o F; ?cancer", "42 y/o F; ?cancer"),
    ("___ y/o; sex not provided", "Age: Not provided; sex not provided"),
])
def test_presentation_redactions_are_explicit_without_resolving_uncertainty(source: str, expected: str) -> None:
    assert normalize_presentation(source) == expected


@pytest.mark.parametrize("value", [None, " ", [], {}])
def test_missing_values_are_not_provided_and_removed_from_catalog(value: JsonValue) -> None:
    with pytest.raises(ValidationError):
        FoundResult(status="found", requested_name="A", matched_key="A", recorded_value=value)
    examinations = {"A": value, "B": 0}
    output = retrieve_tests([ExaminationRequest(name="A", justification="Assess")], examinations, [])
    assert output.test_results[0].status == "not_provided"
    assert available_tests(examinations, output.test_results) == ["B"]


def test_exact_retrieval_preserves_nested_values_and_zero() -> None:
    value: JsonValue = {"units": "mg/dL", "reading": 0, "notes": ["  keep spaces  "]}
    examinations = {"A": value, "Zero": 0}
    output = retrieve_tests([ExaminationRequest(name=name, justification="Assess") for name in examinations], examinations, [])
    assert [item.recorded_value for item in output.test_results if isinstance(item, FoundResult)] == [value, 0]
    assert not output.rejected_requests


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
    output = retrieve_tests([ExaminationRequest(name=requested, justification="Assess")], examinations, [])
    assert not output.rejected_requests
    result = output.test_results[0]
    assert isinstance(result, FoundResult)
    assert result.requested_name == requested
    assert result.matched_key == source
    assert result.recorded_value == value
    assert available_tests(examinations, output.test_results) == ["Other"]
    repeated = retrieve_tests([ExaminationRequest(name=source, justification="Assess")],
                              examinations, output.test_results)
    assert repeated.rejected_requests[0].reason == "already_requested"


def test_trailing_variants_cannot_repeat_missing_tests_or_batch_entries() -> None:
    examinations: dict[str, JsonValue] = {"CT spine ___.": None}
    output = retrieve_tests([
        ExaminationRequest(name="CT spine", justification="Assess"),
        ExaminationRequest(name="ct SPINE.", justification="Assess"),
    ], examinations, [])
    assert output.test_results[0].status == "not_provided"
    assert output.rejected_requests[0].reason == "duplicate_in_batch"
    assert available_tests(examinations, output.test_results) == []
    repeated = retrieve_tests([ExaminationRequest(name="CT spine ___", justification="Assess")],
                              examinations, output.test_results)
    assert repeated.rejected_requests[0].reason == "already_requested"


@pytest.mark.parametrize("requested", ["CT  spine", "CT_spine", "CT spine,", "CT spine_"])
def test_name_normalization_does_not_guess_other_variants(requested: str) -> None:
    output = retrieve_tests([ExaminationRequest(name=requested, justification="Assess")],
                            {"CT spine.": "Recorded"}, [])
    assert not output.test_results
    assert output.rejected_requests[0].reason == "not_in_catalog"


def test_colliding_normalized_keys_fail_without_selecting_a_value() -> None:
    with pytest.raises(ValueError, match="same normalized name"):
        retrieve_tests([ExaminationRequest(name="CT", justification="Assess")],
                       {"CT": "First", "ct.": "Second"}, [])


def test_result_discriminator_rejects_fabricated_value() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ResultUnion).validate_python({"status": "not_provided", "requested_name": "A", "reason": "missing", "recorded_value": "fake"})


def test_adapter_supported_shapes_and_missing_required_fields(tmp_path: Path) -> None:
    raw = {"case_summary": "Synthetic symptoms", "key_pertinent_results_dict": {"A": "value"}, "final_diagnosis": "reference", "stepwise_diagnostic_reasoning_timeline_list": ["private"]}
    path = tmp_path / "synthetic.json"
    for content, expected_id in [([raw], "patient_001"), ({"custom_id": raw}, "custom_id")]:
        path.write_text(json.dumps(content))
        case = load_cases(path)[0]
        assert case.patient_id == expected_id and case.reference.timeline == ["private"]
    path.write_text(json.dumps([{"case_summary": "Synthetic symptoms"}]))
    with pytest.raises(ValidationError):
        load_cases(path)
    path.write_text(json.dumps([dict(raw, patient_id="duplicate"), dict(raw, patient_id="duplicate")]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(path)


def test_concurrent_run_reservation(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(partial(create_run_directory, tmp_path), ["p"] * 8))
    assert len(set(paths)) == 8
    assert sorted(path.name for path in paths) == [f"run{i:03d}" for i in range(1, 9)]


def test_patient_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "other"
    target.mkdir()
    (tmp_path / "p").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, "p")


def test_import_and_constructor_never_import_ml_libraries() -> None:
    code = "from pathlib import Path; from src.model import MedGemmaModel; from src.config import Settings; import sys; m=MedGemmaModel(Path('not-a-model'), Settings()); assert m.model is None; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_every_python_function_has_explicit_types() -> None:
    for path in [*(ROOT / "src").rglob("*.py"), *ROOT.glob("tests/*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is not None, f"{path.name}:{node.name} missing return type"
                for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    if arg.arg not in {"self", "cls"}:
                        assert arg.annotation is not None, f"{path.name}:{node.name}:{arg.arg} missing type"


def test_exact_keys_and_values_survive_adapter_and_retrieval() -> None:
    case = adapt_case(RawCase(case_summary="Synthetic symptoms", key_pertinent_results_dict={"  Blood count  ": "  exact recorded text  "}), "p")
    request_item = ExaminationRequest(name="  Blood count  ", justification="Assess")
    result = retrieve_tests([request_item], case.examinations, []).test_results[0]
    assert isinstance(result, FoundResult)
    assert result.requested_name == result.matched_key == "  Blood count  "
    assert result.recorded_value == "  exact recorded text  "


@pytest.mark.parametrize("layout", ["list", "mapping", "numeric", "fallback"])
def test_load_one_patient(tmp_path: Path, layout: str) -> None:
    raw = {"case_summary": "Synthetic symptoms", "key_pertinent_results_dict": {"A": "exact value"}}
    if layout == "mapping":
        content = {"other": {"unrelated": "invalid record"}, "selected": raw}
        patient_id = "selected"
    elif layout == "numeric":
        content = [dict(raw, patient_id=42)]
        patient_id = "42"
    elif layout == "fallback":
        content = [raw]
        patient_id = "patient_001"
    else:
        content = [{"patient_id": "other"}, dict(raw, patient_id="selected")]
        patient_id = "selected"
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(content))
    case = load_case(path, patient_id)
    assert case.patient_id == patient_id and case.examinations == {"A": "exact value"}


def test_single_patient_missing_duplicate_and_invalid_record(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps([{"patient_id": "p"}]))
    with pytest.raises(LookupError, match="not found"):
        load_case(path, "missing")
    with pytest.raises(ValueError, match="blank"):
        load_case(path, " ")
    with pytest.raises(ValidationError):
        load_case(path, "p")
    path.write_text(json.dumps([{"patient_id": "p"}, {"patient_id": "p"}]))
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
    payload = json.loads(FINAL)
    payload.update(changes)
    with pytest.raises(ValidationError, match=message):
        DiagnosisOutput.model_validate(payload)


@pytest.mark.parametrize("field", ["hypotheses", "current_diagnosis", "action", "requests"])
def test_final_diagnosis_requires_the_same_four_fields(field: str) -> None:
    payload = json.loads(FINAL)
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


