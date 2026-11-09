"""Describe interactive stops without exposing provider payloads or local paths."""

import re

from .llm import LLMBudgetError, LLMNetworkError, LLMOutputError, LLMQuotaError

KEY_NAMES = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}
UNKNOWN_ERROR = "Unknown error — see Manage app logs"


def error_details(error, keys=()):
    if error is None:
        return 'Unknown', None, UNKNOWN_ERROR
    message = getattr(error, 'api_message', None) or str(error)
    if isinstance(error, LLMOutputError):
        message = message.split('\nLast raw text:', 1)[0]
    for key in keys:
        if key:
            message = message.replace(key, '[REDACTED]')
    message = re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|key)=)[^\s&,\x27\x22]+', r'\1[REDACTED]', message)
    message = re.sub(r'(?:AIza[A-Za-z0-9_-]{20,}|(?:gsk_|sk-)[A-Za-z0-9_-]{12,})', '[REDACTED]', message)
    message = re.sub(r'(?i)(?:[A-Z]:\\|/(?:tmp|var/tmp)/)[^\s\x27\x22]+', '[local path]', message)
    status = getattr(error, 'status', None)
    if status is None:
        match = re.search(r'\b(400|401|403|413|429|503|504)\b', message)
        status = int(match[1]) if match else None
    return getattr(error, 'error_class', None) or type(error).__name__, status, " ".join(message.split())[:300] or UNKNOWN_ERROR


def describe_failure(error):
    _, status, message = error_details(error)
    lowered = (getattr(error, 'api_message', None) or str(error)).lower()
    if status == 400 and ('v1beta' in lowered or 'developer instruction' in lowered):
        return "Gemini API version (400)", "The Gemini API rejected the system instruction.", "Gemini library/API version mismatch (fixed in this version; reboot app)."
    if status in (400, 403) and ('api_key_invalid' in lowered or 'permission_denied' in lowered):
        return f"Authentication ({status})", "Gemini rejected the key or permission.", "Key is wrong or not enabled for Gemini API."
    if status == 429 and 'resource_exhausted' in lowered:
        return "Quota exhausted (429)", "The provider reported RESOURCE_EXHAUSTED.", "Daily/minute quota used up. A new key in the SAME Google project shares the quota — create the key in a NEW project."
    if status == 413:
        hint = ("Request too big for Groq free tier; tool now sends one episode per call."
                if getattr(error, 'provider', None) in (None, 'groq') else "Request too large; use a smaller input or another provider.")
        return "Request too large (413)", "The request exceeded the provider size limit.", hint
    if isinstance(error, LLMQuotaError):
        return "Daily quota", "The provider's daily quota is exhausted.", "Switch providers, paste your own key, or resume after the quota resets."
    if status == 429:
        return "Per-minute rate limit (429)", "The provider is limiting requests or tokens per minute.", "Click Find promos again in about one minute. Finished episodes are kept."
    if status in (503, 504) or "timeout" in str(error).lower():
        return f"Provider busy{f' ({status})' if status else ''}", "The provider is busy or timed out.", "Provider busy; press Resume (Find promos), or switch providers."
    if status in (401, 403):
        return f"Authentication ({status})", "The provider rejected the credentials.", "Paste a valid key for this provider in API keys, then click Find promos."
    if isinstance(error, LLMBudgetError):
        return "Call budget", "This attempt reached the call limit.", "Increase Max API calls within your limit, or click Find promos to resume."
    if isinstance(error, LLMNetworkError):
        return "Network", "The provider could not be reached.", "Check the connection or switch providers, then try again."
    if isinstance(error, LLMOutputError):
        return "Invalid model output", "The provider response did not match the requested format.", "Retry or switch providers. Finished episodes are kept."
    return "Stage error", UNKNOWN_ERROR if message == UNKNOWN_ERROR else "The stage could not finish with the available inputs.", "Retry or switch providers. Finished episodes are kept."
