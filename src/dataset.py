"""Unverified DiagBench shape assumptions live here; importing never reads data."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from src.schemas import ExactKey, PatientCase, ReferenceData, Text


class RawCase(BaseModel):
    """Expected relevant fields; unrelated raw fields are intentionally ignored."""
    model_config = ConfigDict(extra="ignore")
    patient_id: str | int | None = None
    case_summary: Text
    key_pertinent_results_dict: dict[ExactKey, JsonValue]
    stepwise_diagnostic_reasoning_timeline_list: JsonValue = None
    final_diagnosis: JsonValue = None
    diagnosis_results: JsonValue = None


def reference_strings(value: JsonValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in reference_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in reference_strings(item)]
    return []


def leakage_flags(summary: str, reference: ReferenceData) -> list[str]:
    """Conservative heuristic, not proof of leakage-free text. Flags block execution."""
    flags: list[str] = []
    if re.search(r"\b(final diagnosis|discharge diagnos\w*|diagnosed with|diagnosis\s*:|confirmed\s+\w+)", summary, re.I):
        flags.append("Presentation contains a diagnostic conclusion marker; manual review required.")
    terms = reference_strings(reference.final_diagnosis) + reference_strings(reference.diagnosis_results)
    if any(len(term.strip()) >= 4 and re.search(r"(?<!\w)" + re.escape(term.strip()) + r"(?!\w)", summary, re.I) for term in terms):
        flags.append("Presentation overlaps reference diagnosis text; manual review required.")
    return flags


def adapt_case(raw: RawCase, fallback_id: str) -> PatientCase:
    reference = ReferenceData(final_diagnosis=raw.final_diagnosis, diagnosis_results=raw.diagnosis_results,
                              timeline=raw.stepwise_diagnostic_reasoning_timeline_list)
    return PatientCase(patient_id=str(raw.patient_id) if raw.patient_id is not None else fallback_id,
                       presentation=raw.case_summary, examinations=raw.key_pertinent_results_dict,
                       reference=reference, leakage_flags=leakage_flags(raw.case_summary, reference))


def _read_records(path: Path) -> list[tuple[str, JsonValue]]:
    """Read JSON once and pair each raw record with its fallback patient ID."""
    content = TypeAdapter(JsonValue).validate_python(json.loads(path.read_text(encoding="utf-8")))
    if isinstance(content, list):
        return [(f"patient_{index:03d}", item) for index, item in enumerate(content, 1)]
    if isinstance(content, dict):
        return list(content.items())
    raise ValueError("Expected a record list or a patient-ID-keyed object; verify the real dataset format")


def load_cases(path: Path) -> list[PatientCase]:
    """Load and validate all patients, rejecting duplicate patient IDs."""
    cases = [adapt_case(RawCase.model_validate(item), fallback_id)
             for fallback_id, item in _read_records(path)]
    ids = [case.patient_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs in dataset")
    return cases


def load_case(path: Path, patient_id: str) -> PatientCase:
    """Return only the requested patient, including any leakage flags.

    JSON is still read in full, but only the selected record is adapted/validated.
    Unknown IDs raise LookupError; duplicate matches raise ValueError.
    """
    patient_id = patient_id.strip()
    if not patient_id:
        raise ValueError("Patient ID cannot be blank")
    matches: list[tuple[str, JsonValue]] = []
    for fallback_id, item in _read_records(path):
        raw_id = item.get("patient_id") if isinstance(item, dict) else None
        effective_id = (fallback_id if raw_id is None else str(raw_id)).strip()
        if effective_id == patient_id:
            matches.append((fallback_id, item))
    if not matches:
        raise LookupError(f"Patient ID {patient_id!r} was not found")
    if len(matches) > 1:
        raise ValueError(f"Duplicate patient ID: {patient_id!r}")
    fallback_id, item = matches[0]
    return adapt_case(RawCase.model_validate(item), fallback_id)
