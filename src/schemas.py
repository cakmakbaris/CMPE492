"""Validated boundaries; model proposals are deliberately separate from retrieved data."""
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# Catalog keys must remain byte-for-byte as supplied, including surrounding spaces.
ExactKey = Annotated[str, StringConstraints(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: Text


class GenerateRequest(StrictModel):
    prompt: Text | None = Field(default=None, max_length=100000)
    messages: list[ChatMessage] | None = Field(default=None, min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=3072)
    output_hidden_states: bool = False
    capture_activations: bool = Field(default=False, description="Text-only Qwen3.5/MedGemma endpoint capture: all blocks and final norm at last input/output tokens, input embedding and token log probabilities; separate v3 directory.")
    stream: bool = False
    capture_diagnosis_tokens: bool = False
    response_schema: dict[str, JsonValue] | None = Field(default=None, min_length=1,
        description="Optional JSON Schema constrained by XGrammar on the non-thinking Qwen server; Python semantic validation remains required.")
    steering_alpha: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False,
                                         description="Experimental D82: last input token, block 22, once per call; server must have an explicit steering bundle.")

    @model_validator(mode="after")
    def valid_conversation(self) -> Self:
        if self.capture_diagnosis_tokens and (self.capture_activations or self.output_hidden_states or self.steering_alpha is not None):
            raise ValueError("Diagnosis-span capture must be used separately")
        if self.output_hidden_states and self.capture_activations:
            raise ValueError("Choose full hidden states or endpoint activations, not both")
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or messages")
        if self.messages is not None:
            turns = self.messages[1:] if self.messages[0].role == "system" else self.messages
            if not turns or turns[-1].role != "user":
                raise ValueError("Conversation must end with a user message")
            for index, message in enumerate(turns):
                if message.role != ("user" if index % 2 == 0 else "assistant"):
                    raise ValueError("After an optional initial system message, alternate user/assistant turns")
        return self

    def conversation(self) -> list[ChatMessage]:
        if self.messages is not None:
            return self.messages
        assert self.prompt is not None
        return [ChatMessage(role="user", content=self.prompt)]


class SteeringAudit(StrictModel):
    alpha: float
    bundle_sha256: str
    module: str = "model.layers.21"
    input_length: int
    prefill_visits: int
    patch_count: int
    sigma: float
    before_norm: float
    actual_delta_norm: float
    relative_delta_norm: float


class GenerateResponse(StrictModel):
    request_id: str
    text: str
    generated_tokens: int
    generation_seconds: float
    hidden_states_path: str | None = None
    activations_path: str | None = None
    steering: SteeringAudit | None = None


class GenerateChunk(StrictModel):
    kind: Literal["chunk"] = "chunk"
    text: str


class GenerateDone(StrictModel):
    kind: Literal["completed"] = "completed"
    response: GenerateResponse


class GenerateError(StrictModel):
    kind: Literal["error"] = "error"
    error_type: str


GenerateStreamEvent = Annotated[GenerateChunk | GenerateDone | GenerateError, Field(discriminator="kind")]


class ReferenceData(StrictModel):
    """Evaluation-only information. Never serialized into diagnosis prompts."""
    final_diagnosis: JsonValue = None
    diagnosis_results: JsonValue = None
    timeline: JsonValue = None


class PatientCase(StrictModel):
    patient_id: Text
    presentation: Text
    examinations: dict[ExactKey, JsonValue]
    reference: ReferenceData = Field(default_factory=ReferenceData)
    leakage_flags: list[str] = Field(default_factory=list)


# Field descriptions are included in model_json_schema() sent to Diagnosis.
class DiagnosisHypothesis(StrictModel):
    diagnosis: Text = Field(
        description='Name of one possible diagnosis in the current differential; this is a hypothesis, not a confirmed finding.'
    )
    reasoning: Text = Field(
        description='Free-text evidence-based clinical rationale explaining why this hypothesis is proposed or retained, which supplied findings support it, material contradictions or uncertainty, and evidence-based changes from prior assessments. Usually 3–5 focused sentences; fewer when sufficient.'
    )


class TestRequest(StrictModel):
    name: ExactKey = Field(
        description='Copy one name exactly from the current available_tests list. Python tolerates only trailing periods, whitespace and ___ markers. Do not invent names or reuse earlier catalogs.'
    )
    justification: Text = Field(
        description='Brief explanation of what diagnostic question this examination would help resolve.'
    )


class DiagnosisOutput(StrictModel):
    """One diagnosis response shape for every round, including finalization."""
    hypotheses: list[DiagnosisHypothesis] = Field(
        min_length=1,
        description="Current nonempty differential with unique diagnosis names and brief evidence-based reasoning. Update it each round, including finalization.",
    )
    current_diagnosis: Text = Field(
        description="The name of the most likely hypothesis in this round, copied exactly from one hypotheses entry's diagnosis field. It is provisional until finalization is accepted.",
    )
    action: Literal["request_tests", "final_diagnosis", "insufficient_information"] = Field(
        description="Request evidence, submit the supported leading diagnosis, or stop incomplete when no conclusion is supported and no useful test remains.",
    )
    requests: list[TestRequest] = Field(
        max_length=2,
        description="For request_tests, select one or two highest-priority exact catalog entries. Never combine multiple catalog names into one request. Otherwise provide an empty list. Never include predicted results.",
    )

    @model_validator(mode="after")
    def consistent_diagnosis(self) -> Self:
        names = [hypothesis.diagnosis for hypothesis in self.hypotheses]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Hypothesis diagnosis names must be unique")
        if self.current_diagnosis not in names:
            raise ValueError("current_diagnosis must exactly match a hypothesis diagnosis name")
        if self.action == "request_tests" and not self.requests:
            raise ValueError("request_tests requires at least one examination request")
        if self.action != "request_tests" and self.requests:
            raise ValueError("A terminal action requires an empty requests list")
        return self


class FoundResult(StrictModel):
    """Python-retrieved exact value. Null, blank text, and empty containers are forbidden."""
    status: Literal["found"] = Field(
        description='Python has retrieved an existing, non-missing recorded value for the matched examination.'
    )
    requested_name: ExactKey = Field(
        description='The examination name originally requested by the Diagnosis Agent.'
    )
    matched_key: ExactKey = Field(
        description='The exact patient examination dictionary key validated by Python.'
    )
    recorded_value: JsonValue = Field(
        description='The exact JSON value retrieved from the patient record by Python; never generated or completed by an LLM.'
    )

    @model_validator(mode="after")
    def present_value(self) -> Self:
        if is_missing(self.recorded_value):
            raise ValueError("A found result requires a recorded value")
        return self


class NotProvidedResult(StrictModel):
    """Missing evidence, never interpreted as a normal finding. Value fields are forbidden."""
    status: Literal["not_provided"] = Field(
        description='No relevant recorded value was provided; this is missing evidence, not a normal finding or a runtime error.'
    )
    requested_name: ExactKey = Field(
        description='The examination name originally requested by the Diagnosis Agent.'
    )
    reason: Text = Field(
        description='The selected catalog entry has an empty recorded value. Invalid or repeated requests are rejected separately.'
    )


TestResult = Annotated[FoundResult | NotProvidedResult, Field(discriminator="status")]


def is_missing(value: JsonValue) -> bool:
    """Null and empty containers/text are missing; numeric zero is a real result."""
    return value is None or value == [] or value == {} or isinstance(value, str) and not value.strip()


class DiagnosisInput(StrictModel):
    """Current catalog/results snapshot; no unrequested values or reference data."""
    presentation: Text
    available_tests: list[ExactKey]
    test_results: list[TestResult]
    feedback: str
    round: int
    max_rounds: int


class RejectedTestRequest(StrictModel):
    requested_name: ExactKey
    reason: Literal["already_requested", "not_in_catalog", "duplicate_in_batch"]


class RetrievalOutput(StrictModel):
    test_results: list[TestResult]
    rejected_requests: list[RejectedTestRequest]
