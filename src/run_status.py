"""Describe interactive stops without exposing provider payloads or local paths."""

from .llm import LLMBudgetError, LLMNetworkError, LLMOutputError, LLMQuotaError

KEY_NAMES = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}


def describe_failure(error):
    status = getattr(error, "status", None)
    if isinstance(error, LLMQuotaError):
        return "Daily quota", "The provider's daily quota is exhausted.", "Switch providers, paste your own key, or resume after the quota resets."
    if status == 429:
        return "Per-minute rate limit (429)", "The provider is limiting requests or tokens per minute.", "Click Find promos again in about one minute. Finished episodes are kept."
    if status in (503, 504) or "timeout" in str(error).lower():
        return f"Provider busy{f' ({status})' if status else ''}", "The provider is busy or timed out.", "Retry in a moment or switch providers."
    if status in (401, 403):
        return f"Authentication ({status})", "The provider rejected the credentials.", "Paste a valid key for this provider in API keys, then click Find promos."
    if isinstance(error, LLMBudgetError):
        return "Call budget", "This attempt reached the call limit.", "Increase Max API calls within your limit, or click Find promos to resume."
    if isinstance(error, LLMNetworkError):
        return "Network", "The provider could not be reached.", "Check the connection or switch providers, then try again."
    if isinstance(error, LLMOutputError):
        return "Invalid model output", "The provider response did not match the requested format.", "Retry or switch providers. Finished episodes are kept."
    return "Stage error", "The stage could not finish with the available inputs.", "Retry or switch providers. Finished episodes are kept."
