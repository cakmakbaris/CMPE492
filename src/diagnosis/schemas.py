"""Output contracts for all-tests and stepwise diagnosis."""
from typing import Literal, Self

from pydantic import Field, model_validator

from src.schemas import DiagnosisHypothesis, StrictModel, Text


class StepwiseTestRequest(StrictModel):
    test_id: Text
    reason: Text


class StepwiseDiagnosisOutput(StrictModel):
    hypotheses: list[DiagnosisHypothesis] = Field(min_length=1, max_length=5)
    action: Literal["request_tests", "final_diagnosis"]
    requests: list[StepwiseTestRequest] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_unique_items(self) -> Self:
        diagnoses = {item.diagnosis.casefold() for item in self.hypotheses}
        test_ids = {item.test_id for item in self.requests}
        if len(diagnoses) != len(self.hypotheses) or len(test_ids) != len(self.requests):
            raise ValueError("Diagnoses and requested test IDs must each be distinct")
        if bool(self.requests) != (self.action == "request_tests"):
            raise ValueError("request_tests requires at least one request and final_diagnosis requires none")
        return self


class BaselineDiagnosisOutput(StrictModel):
    hypotheses: list[DiagnosisHypothesis] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_diagnoses(self) -> Self:
        names = [h.diagnosis.casefold() for h in self.hypotheses]
        if len(set(names)) != len(names):
            raise ValueError("Hypotheses must be distinct")
        return self


class EvidenceHypothesis(StrictModel):
    diagnosis: Text
    supporting_findings: list[Text] = Field(min_length=1)
    contradicting_findings: list[Text]


class EvidenceDiagnosisOutput(StrictModel):
    hypotheses: list[EvidenceHypothesis] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_diagnoses(self) -> Self:
        names = [hypothesis.diagnosis.casefold() for hypothesis in self.hypotheses]
        if len(set(names)) != len(names):
            raise ValueError("Hypotheses must be distinct")
        return self
