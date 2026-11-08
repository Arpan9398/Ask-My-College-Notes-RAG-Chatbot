"""LLM provider factory and translation of API failures into readable messages."""
from __future__ import annotations

import os

from .config import Settings


class LLMError(Exception):
    """Raised with a user-friendly message when the LLM cannot be used."""


def build_llm(settings: Settings):
    provider = settings.llm_provider
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, add your key, "
                "and restart the app."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LLMError("Missing dependency 'langchain-anthropic'. Run: pip install -r requirements.txt") from exc
        return ChatAnthropic(
            model=settings.anthropic_model, temperature=0, max_tokens=1024, timeout=60, max_retries=2
        )
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, "
                "and restart the app."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LLMError("Missing dependency 'langchain-openai'. Run: pip install -r requirements.txt") from exc
        return ChatOpenAI(model=settings.openai_model, temperature=0, timeout=60, max_retries=2)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic' or 'openai' in your .env file.")


def explain_llm_error(exc: Exception) -> str:
    """Map provider exceptions (Anthropic/OpenAI/httpx) to a short, actionable message."""
    if isinstance(exc, LLMError):
        return str(exc)
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    detail = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else name

    if status == 401 or "Authentication" in name:
        return "The LLM API rejected your API key (401). Check the key in your .env file."
    if status == 403 or "PermissionDenied" in name:
        return "Your API key does not have permission to use this model (403)."
    if status == 404 or "NotFound" in name:
        return "The configured model was not found (404). Check ANTHROPIC_MODEL / OPENAI_MODEL in .env."
    if status == 429 or "RateLimit" in name:
        return "The LLM API rate limit or quota was hit (429). Wait a moment and try again, or check your billing."
    if status is not None and status >= 500:
        return f"The LLM provider had a server error ({status}). Please try again shortly."
    if "Timeout" in name:
        return "The LLM request timed out. Please try again."
    if "Connection" in name:
        return "Could not reach the LLM API. Check your internet connection."
    return f"The LLM request failed ({name}): {detail}"
