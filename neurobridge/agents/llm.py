"""Optional LLM assistance (Anthropic Claude). The LLM is used only for language tasks: decomposing a question
into sub-questions/queries and drafting a reviewer narrative. It never produces statistics, and every LLM output
is validated against a schema and intersected with allowed values before use. Without a provider configured,
NeuroBridge runs fully deterministically."""

from __future__ import annotations

from typing import Optional, TypeVar

from pydantic import BaseModel

from neurobridge.config import REPO_ROOT, get_settings
from neurobridge.logging_utils import get_logger, log_event

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


def provider_status() -> dict:
    s = get_settings()
    if s.llm_provider == "none":
        return {"provider": "none", "available": False, "reason": "NEUROBRIDGE_LLM_PROVIDER=none (deterministic mode)"}
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return {"provider": s.llm_provider, "available": False, "reason": "anthropic package not installed (pip install '.[llm]')"}
    return {"provider": s.llm_provider, "available": True, "model": s.llm_model}


def structured(system: str, prompt: str, schema: type[T], max_tokens: int = 4000) -> T:
    """One structured-output call to Claude, validated against `schema`. Raises LLMUnavailable on any failure."""
    s = get_settings()
    if s.llm_provider != "anthropic":
        raise LLMUnavailable("no LLM provider configured")
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable("anthropic SDK not installed") from exc
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)  # makes ANTHROPIC_API_KEY from .env visible to the SDK
        client = anthropic.Anthropic()
        resp = client.messages.parse(
            model=s.llm_model,
            max_tokens=max_tokens,
            system=system,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("Anthropic authentication failed (set ANTHROPIC_API_KEY or run `ant auth login`)") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("Anthropic rate limit reached") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Anthropic API error {exc.status_code}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable("could not reach the Anthropic API") from exc
    if resp.stop_reason == "refusal":
        raise LLMUnavailable("model declined the request")
    parsed: Optional[T] = resp.parsed_output
    if parsed is None:
        raise LLMUnavailable("no parsable structured output")
    log_event(log, "llm.structured", model=s.llm_model, schema=schema.__name__, request_id=getattr(resp, "_request_id", None))
    return parsed
