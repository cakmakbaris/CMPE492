"""Catalog retrieval ignoring letter case and trailing formatting, never synonyms."""
import re

from pydantic import JsonValue

from src.schemas import (FoundResult, NotProvidedResult, RejectedTestRequest,
                         RetrievalOutput, TestRequest, TestResult, is_missing)


def normalize_test_name(name: str) -> str:
    """Ignore letter case, trailing whitespace, periods and triple underscores."""
    return re.sub(r"(?:\s+|\.|_{3})+$", "", name).casefold()


def available_tests(examinations: dict[str, JsonValue], test_results: list[TestResult]) -> list[str]:
    """Preserve source names while removing all normalized forms of consumed tests."""
    used = {normalize_test_name(result.requested_name) for result in test_results}
    return [name for name in examinations if normalize_test_name(name) not in used]


def retrieve_tests(requests: list[TestRequest], examinations: dict[str, JsonValue],
                   test_results: list[TestResult]) -> RetrievalOutput:
    catalog: dict[str, str] = {}
    for key in examinations:
        normalized = normalize_test_name(key)
        if not normalized:
            continue
        if normalized in catalog:
            raise ValueError("Multiple examination keys have the same normalized name")
        catalog[normalized] = key
    used = {normalize_test_name(result.requested_name) for result in test_results}
    seen: set[str] = set()
    output = RetrievalOutput(test_results=[], rejected_requests=[])
    for request in requests:
        name = request.name
        normalized = normalize_test_name(name)
        reason = ("duplicate_in_batch" if normalized in seen else
                  "already_requested" if normalized in used else
                  "not_in_catalog" if normalized not in catalog else None)
        seen.add(normalized)
        if reason is not None:
            output.rejected_requests.append(RejectedTestRequest(requested_name=name, reason=reason))
            continue
        key = catalog[normalized]
        value = examinations[key]
        if is_missing(value):
            output.test_results.append(NotProvidedResult(
                status="not_provided", requested_name=name,
                reason="The catalog entry exists, but its recorded value is missing. Do not request it again."))
        else:
            output.test_results.append(FoundResult(status="found", requested_name=name,
                                                   matched_key=key, recorded_value=value))
    return output
