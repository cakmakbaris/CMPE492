"""Streaming client for the persistent loopback model server."""
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import JsonValue, TypeAdapter

from src.config import Settings
from src.schemas import ChatMessage, GenerateDone, GenerateError, GenerateRequest, GenerateStreamEvent
from src.utils import append_model_text


class HttpModel:
    """Send agent prompts to the existing local server instead of loading another model."""

    def __init__(self, url: str, settings: Settings, capture_activations: bool = False,
                 steering_alpha: float | None = None, steering_sha256: str | None = None) -> None:
        address = urlsplit(url)
        if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Provide an HTTP model server on the loopback interface")
        self.endpoint = url.rstrip("/") + "/generate"
        self.settings = settings
        self.capture_activations = capture_activations
        self.steering_alpha = steering_alpha
        self.steering_sha256 = steering_sha256
        self.output_log: Path | None = None
        self.call_count = 0

    def set_output_log(self, path: Path) -> None:
        path.open("x", encoding="utf-8").close()
        self.output_log = path

    def generate(self, prompt: str) -> str:
        return self.generate_messages([ChatMessage(role="user", content=prompt)])

    def generate_messages(self, messages: list[ChatMessage],
                          response_schema: dict[str, JsonValue] | None = None,
                          capture_diagnosis_tokens: bool = False) -> str:
        proposal = GenerateRequest(messages=messages, max_new_tokens=self.settings.max_new_tokens,
                                   output_hidden_states=False, capture_activations=self.capture_activations,
                                   stream=True, steering_alpha=self.steering_alpha,
                                   response_schema=response_schema, capture_diagnosis_tokens=capture_diagnosis_tokens)
        request = Request(self.endpoint, data=proposal.model_dump_json().encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        # The server may spend minutes generating a full agent JSON response.
        # There are no hidden retries or fallback local model loads.
        self.call_count += 1
        if self.output_log is not None:
            append_model_text(self.output_log, f"\n=== Call {self.call_count} | "
                              f"{datetime.now(timezone.utc).isoformat()} ===\n")
        result = None
        adapter = TypeAdapter(GenerateStreamEvent)
        try:
            with urlopen(request, timeout=600) as response:
                if "application/x-ndjson" not in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Server does not support streaming; restart the updated server")
                for line in response:
                    event = adapter.validate_json(line)
                    if isinstance(event, GenerateError):
                        raise RuntimeError(f"Model server generation failed: {event.error_type}")
                    if isinstance(event, GenerateDone):
                        result = event.response
                    elif self.output_log is not None:
                        append_model_text(self.output_log, event.text)
            if result is None:
                raise RuntimeError("Model stream ended without a completion record")
            if self.capture_activations or capture_diagnosis_tokens:
                if result.activations_path is None:
                    raise RuntimeError("Server returned no compact activations; restart the updated server")
                if self.output_log is not None:
                    with (self.output_log.parent / "activations.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"call": self.call_count, "request_id": result.request_id,
                                                "activations_path": result.activations_path,
                                                "generated_tokens": result.generated_tokens}) + "\n")
        except Exception as exc:
            if self.output_log is not None:
                append_model_text(self.output_log, f"\n[Call failed: {type(exc).__name__}: {exc}]\n")
            raise
        if self.output_log is not None:
            if self.steering_alpha is not None:
                if (result.steering is None or result.steering.alpha != self.steering_alpha
                        or result.steering.bundle_sha256 != self.steering_sha256
                        or result.steering.prefill_visits != 1
                        or result.steering.patch_count != int(self.steering_alpha > 0)):
                    raise RuntimeError("Missing or mismatched steering audit; stop the pilot")
                with (self.output_log.parent / "steering_calls.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"call": self.call_count,
                        "messages": [message.model_dump() for message in messages],
                        "response": result.model_dump(mode="json")}, ensure_ascii=False) + "\n")
            append_model_text(self.output_log, f"\n[Completed: request_id={result.request_id}; "
                              f"tokens={result.generated_tokens}; limit={self.settings.max_new_tokens}; "
                              f"seconds={result.generation_seconds:.1f}]\n")
        print(f"Model request {result.request_id}: {result.generated_tokens} tokens, "
              f"{result.generation_seconds:.1f}s", flush=True)
        return result.text
