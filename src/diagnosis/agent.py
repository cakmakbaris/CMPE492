"""Diagnosis only sees the explicitly projected input, never the raw patient case."""
from json import dumps
from pathlib import Path
from typing import Protocol

from src.schemas import ChatMessage, DiagnosisInput, DiagnosisOutput
from src.utils import extract_final_json, normalize_presentation


class JsonModel(Protocol):
    def generate_messages(self, messages: list[ChatMessage]) -> str: ...


PROMPT = (Path(__file__).resolve().parents[2] / "prompts/diagnosis/agent.txt").read_text(encoding="utf-8")


def diagnosis_messages(history: list[DiagnosisOutput], context: DiagnosisInput) -> list[ChatMessage]:
    """Rebuild alternating turns with one current results/catalog snapshot per call."""
    messages = [ChatMessage(role="system", content=PROMPT.format(
        schema=dumps(DiagnosisOutput.model_json_schema())))]
    context = context.model_copy(update={"presentation": normalize_presentation(context.presentation)})
    if history:
        messages.append(ChatMessage(role="user", content=dumps({"presentation": context.presentation})))
        for index, output in enumerate(history):
            if index:
                messages.append(ChatMessage(role="user", content=
                    "Historical assessment. All collected evidence is consolidated in the latest test_results; "
                    "only the latest available_tests list is valid."))
            messages.append(ChatMessage(role="assistant", content=output.model_dump_json()))
    snapshot = context.model_dump_json(exclude={"presentation"} if history else set())
    instruction = (
        "**Select tests ONLY from this turn's `available_tests`: at most TWO exact names.**\n"
        "Update your assessment using all `test_results` and feedback; return the next Diagnosis JSON."
    )
    if context.round > 1:
        instruction = (
            "Reassess using all collected `test_results` and feedback.\n"
            "Retain, revise, or remove previous hypotheses as warranted; "
            "add new hypotheses only when supported.\n"
            "Follow the system instructions.\n"
            "**If requesting tests, request at most TWO entries. "
            "Each `request.name` MUST exactly match an entry in the CURRENT `available_tests`.**\n"
            "Return the next Diagnosis JSON."
        )
    messages.append(ChatMessage(role="user", content=instruction + "\n\n```json\n" + snapshot + "\n```"))
    return messages


def diagnose(model: JsonModel, messages: list[ChatMessage]) -> DiagnosisOutput:
    return DiagnosisOutput.model_validate_json(extract_final_json(
        model.generate_messages(messages)))
