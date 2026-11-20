"""Small filesystem helpers. Logs can contain patient information; keep them local."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4


class FinalJsonError(ValueError):
    """The final response is missing or is not a standalone JSON object."""


def normalize_presentation(presentation: str) -> str:
    """Clarify redacted text for prompts without changing recorded source data."""
    age_pattern = r"_{3,}\s*(?:y\s*/\s*o|years?\s+old|years?-old|yo\b)"
    for abbreviation, label in (("F", "Female"), ("M", "Male")):
        presentation = re.sub(age_pattern + rf"\s+{abbreviation}\b",
                              f"{label}. Age: Not provided", presentation, flags=re.IGNORECASE)
    presentation = re.sub(age_pattern, "Age: Not provided", presentation, flags=re.IGNORECASE)
    return re.sub(r"_{3,}", "Not provided", presentation)


def extract_final_json(text: str) -> str:
    """Select the final-answer segment; leave JSON/schema validation to Pydantic."""
    final = text.rsplit("<unused95>", 1)[-1].strip()
    if final.startswith("```"):
        fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n\s*```", final, re.IGNORECASE)
        if fenced is None:
            raise FinalJsonError("Final JSON fence is incomplete or contains surrounding text")
        final = fenced.group(1).strip()
    if not final.startswith("{") or not final.endswith("}"):
        raise FinalJsonError("Expected a complete final JSON object")
    return final


def create_request_directory(root: Path) -> Path:
    """Reserve a fresh request folder; never reuse or overwrite an existing one."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    return path


def patient_directory_name(patient_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", patient_id).strip("_")[:80] or "patient"
    # Preserve ordinary IDs; disambiguate IDs changed by sanitization/truncation.
    if safe != patient_id:
        safe += "-" + hashlib.sha256(patient_id.encode()).hexdigest()[:12]
    return safe


def create_run_directory(root: Path, patient_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    patient = root / patient_directory_name(patient_id)
    if patient.is_symlink():
        raise ValueError("Patient directory cannot be a symbolic link")
    patient.mkdir(exist_ok=True)
    numbers = [int(path.name[3:]) for path in patient.iterdir() if re.fullmatch(r"run\d+", path.name)]
    number = max(numbers, default=0) + 1
    while True:
        path = patient / f"run{number:03d}"
        try:
            path.mkdir()  # Atomic reservation: concurrent executions never overwrite each other.
            return path
        except FileExistsError:
            number += 1


def append_model_text(path: Path, text: str) -> None:
    """Append and close each chunk so a separate terminal can follow the raw output."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replace_json(path: Path, value: object) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
