"""Shared, cached LLM calls with small swappable provider adapters."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from .models import Episode
from .render import render_episode
from .timeutils import format_timestamp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
Model = TypeVar("Model", bound=BaseModel)
USAGE_FIELDS = [
    "time", "tag", "provider", "model", "cached", "latency_s",
    "in_tokens", "out_tokens", "ok", "error",
]


class LLMRequestError(RuntimeError):
    """A provider could not complete a request."""

    def __init__(self, message, *, status=None, provider=None, model=None):
        super().__init__(message)
        self.status = status
        self.provider = provider
        self.model = model


class LLMBatchSplitError(LLMRequestError):
    """A busy five-episode Gemini request should be retried in smaller groups."""


class LLMQuotaError(LLMRequestError):
    """The provider's daily quota is exhausted, rather than temporarily busy."""


class LLMBudgetError(LLMRequestError):
    """The run has used its configured real-attempt budget."""


class LLMNetworkError(LLMRequestError):
    """Three consecutive attempts failed without a provider response."""


class LLMRateLimitError(LLMRequestError):
    """The current run exhausted its allowed wait for rate limits."""


NETWORK_BLOCKED = "network blocked — approve network or run locally"


@dataclass
class CallBudget:
    """Share an attempt counter across models used by one run."""

    limit: int = 14
    used: int = 0
    connection_errors: int = 0
    consecutive_connection_errors: int = 0
    retries: int = 0
    exempt_minute_limits: bool = False
    wait_limit: float | None = None
    waited: float = 0
    progress: Callable | None = None
    token_history: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.limit < 0:
            raise ValueError("max_api_calls must be non-negative")

    def check(self):
        if self.consecutive_connection_errors >= 3:
            raise LLMNetworkError(NETWORK_BLOCKED)
        if self.used >= self.limit:
            raise LLMBudgetError(f"API call budget reached ({self.used}/{self.limit}); progress saved")

    def consume(self):
        self.check()
        self.used += 1

    def wait_for_rate_limit(self, provider, model, seconds):
        if self.wait_limit is not None and self.waited + seconds > self.wait_limit:
            raise LLMRateLimitError("Rate-limit wait allowance exhausted.", status=429, provider=provider, model=model)
        self.waited += seconds
        remaining = seconds
        while remaining > 0:
            if self.progress:
                self.progress(provider, remaining)
            chunk = min(5.0, remaining)
            _retry_sleep(chunk)
            remaining -= chunk


class LLMOutputError(RuntimeError):
    """Model output still failed validation after repair attempts."""

    def __init__(self, tag: str, raw_text: str, error: str):
        self.tag = tag
        self.raw_text = raw_text
        super().__init__(f"[{tag}] Output validation exhausted: {error}\nLast raw text: {raw_text}")


class FormatRejected(LLMRequestError):
    """The provider explicitly rejected the requested structured format."""

    def __init__(self, message: str, raw_text: str = ""):
        self.raw_text = raw_text
        super().__init__(message)


class ExtraValidationError(ValueError):
    """Domain checks rejected otherwise valid JSON."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class Request:
    system: str
    user: str
    temperature: float
    schema: type[BaseModel] | None = None
    structured: bool = True


@dataclass(frozen=True)
class Response:
    text: str
    in_tokens: int | None = None
    out_tokens: int | None = None


class RateLimiter:
    """Space attempts and cap estimated input tokens in a rolling minute."""

    def __init__(self, requests_per_minute: float, clock=None, sleep=None, tokens_per_minute: int | None = None):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.last_start: float | None = None
        if tokens_per_minute is not None and tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive or None")
        self.tokens_per_minute = tokens_per_minute
        self.token_history: deque[tuple[float, int]] = deque()

    def wait(self, estimated_tokens: int = 0) -> None:
        if self.tokens_per_minute and estimated_tokens > self.tokens_per_minute:
            raise LLMRequestError(
                f"Estimated input ({estimated_tokens} tokens) exceeds tokens_per_minute={self.tokens_per_minute}; "
                "reduce the prompt or configure the provider's actual limit."
            )
        now = self.clock()
        if self.last_start is not None:
            delay = self.interval - (now - self.last_start)
            if delay > 0:
                self._sleep(delay)
        if self.tokens_per_minute:
            while True:
                now = self.clock()
                while self.token_history and now - self.token_history[0][0] >= 60:
                    self.token_history.popleft()
                used = sum(tokens for _, tokens in self.token_history)
                if used + estimated_tokens <= self.tokens_per_minute:
                    break
                self._sleep(max(0.001, 60 - (now - self.token_history[0][0])))
        self.last_start = self.clock()
        if self.tokens_per_minute:
            self.token_history.append((self.last_start, estimated_tokens))

    def _sleep(self, seconds: float) -> None:
        if getattr(self, "on_wait", None):
            self.on_wait(seconds)
            return
        while seconds > 0:
            chunk = min(seconds, 30.0)
            self.sleep(chunk)
            seconds -= chunk


def _status(error: Exception) -> int | None:
    value = getattr(error, "status_code", None) or getattr(error, "code", None)
    return value if isinstance(value, int) else None


def _transient(error: Exception) -> bool:
    if _daily_quota(error):
        return False
    status = _status(error)
    if status is not None:
        return status == 429 or 500 <= status <= 599
    name = type(error).__name__.lower()
    return "overloaded" in str(error).lower() or isinstance(error, (TimeoutError, ConnectionError)) or any(
        marker in name for marker in ("timeout", "connectionerror", "connecterror")
    )


def _gemini_busy(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return _status(error) in (503, 504) or any(
        word in message for word in ("overloaded", "timeout", "timed out"))


def _connection_failure(error: Exception) -> bool:
    """Only exempt transport errors with no HTTP response/status."""
    if _status(error) is not None or getattr(error, "response", None) is not None:
        return False
    name = type(error).__name__.lower()
    return isinstance(error, (TimeoutError, ConnectionError)) or any(
        marker in name for marker in ("timeout", "connectionerror", "connecterror", "connecttimeout", "gaierror"))


def _daily_quota(error: Exception) -> bool:
    message = str(error).lower()
    has_daily_marker = any(marker in message for marker in ("per day", "perday", "daily", "requests/day", "tokens/day"))
    has_limit = any(marker in message for marker in ("limit", "quota", "exhaust"))
    return (_status(error) == 429 or "resource_exhausted" in message) and has_daily_marker and has_limit


def _retry_wait(state) -> float:
    """Honor provider Retry-After before falling back to exponential backoff."""
    error = state.outcome.exception()
    if _status(error) == 429:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or getattr(error, "headers", {}) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError):
                    pass
        match = re.search(r"(?:try again|retry) in\s+([\d.]+)\s*(?:seconds?|s\b)", str(error), re.I)
        if match:
            return float(match[1])
    return wait_exponential(multiplier=1, min=1, max=8)(state)


def _retry_sleep(seconds: float) -> None:
    while seconds > 0:
        chunk = min(seconds, 30.0)
        time.sleep(chunk)
        seconds -= chunk


def _json_text(raw: str) -> str:
    """Remove optional fences and prose without touching stored raw output."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    openings = [(text.find(char), char) for char in ("{", "[") if text.find(char) >= 0]
    if not openings:
        return text
    first, opening = min(openings)
    last = text.rfind("}" if opening == "{" else "]")
    if first >= 0 and last >= first:
        return text[first:last + 1]
    return text


def json_user_prompt(user: str, schema: type[BaseModel]) -> str:
    """Use the same schema instructions in planning and actual requests."""
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return user + "\n\nReturn ONLY valid JSON matching this schema:\n" + schema_text


def estimate_json_tokens(system: str, user: str, schema: type[BaseModel]) -> int:
    return math.ceil(len(system + json_user_prompt(user, schema)) / 4)


def request_token_limit(config: dict, provider: str) -> int | None:
    settings = config.get("llm", config)
    limits = [settings.get("max_request_tokens"), settings.get("providers", {}).get(provider, {}).get("max_request_tokens")]
    values = [int(value) for value in limits if value is not None]
    return min(values) if values else None


def step_model_config(config: dict, step: str, provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    settings = config.get("llm", {})
    selected = config.get("models", {}).get(step, {})
    return (provider or selected.get("provider", settings.get("provider", "gemini")),
            model or selected.get("model", settings.get("model", "gemini-2.5-flash")))


class LLMClient:
    """Apply reliability and accounting around a provider's single-call method."""

    provider_name = "base"

    def __init__(
        self, model: str, temperature: float = 0.4, max_retries: int = 3,
        requests_per_minute: float = 8, use_cache: bool = True,
        cache_dir: str | Path = ".cache/llm",
        raw_log_dir: str | Path = "outputs/llm_raw",
        usage_log: str | Path | None = None,
        tokens_per_minute: int | None = None,
        max_request_tokens: int | None = None,
        max_api_calls: int | None = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1 (total attempts)")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_request_tokens = max_request_tokens
        self.call_budget = CallBudget(max_api_calls) if max_api_calls is not None else None
        self.connection_errors = 0
        self.consecutive_connection_errors = 0
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.raw_log_dir = Path(raw_log_dir)
        self.usage_log = Path(usage_log) if usage_log else self.raw_log_dir.parent / "llm_usage.csv"
        self.limiter = RateLimiter(requests_per_minute, tokens_per_minute=tokens_per_minute)
        self.last_call: dict = {}
        self.last_raw_text = ""
        self.schema_fallback_used = False
        self.repair_events: list[dict] = []
        self.quota_error: str | None = None
        self._plain_schemas: set[type[BaseModel]] = set()
        self._memory_cache: dict[str, BaseModel | str] = {}
        self._secrets: list[str] = []

    def _send(self, request: Request) -> Response:
        raise NotImplementedError

    def _safe_error(self, error: Exception) -> str:
        message = f"{type(error).__name__}: {error}"
        for secret in self._secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    def _key(self, system: str, user: str, schema: type[BaseModel] | None, temperature: float) -> str:
        identity = [
            self.provider_name, self.model, temperature, system, user,
            schema.__name__ if schema else None,
        ]
        if getattr(self, "thinking_budget", None) is not None:
            identity.append({"thinking_budget": self.thinking_budget})
        if getattr(self, "reasoning_effort", None) is not None:
            identity.append({"reasoning_effort": self.reasoning_effort})
        encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _usage(self, **values) -> None:
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        exists = self.usage_log.exists() and self.usage_log.stat().st_size > 0
        with self.usage_log.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=USAGE_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({field: values.get(field) for field in USAGE_FIELDS})

    def _raw_log(self, tag: str, key: str, record: dict) -> str:
        """Create immutable per-attempt files, retaining retries separately."""
        self.raw_log_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:80] or "call"
        stem = f"{safe_tag}_{key[:10]}"
        occurrence = 1
        while True:
            suffix = "" if occurrence == 1 else f"_{occurrence:03d}"
            path = self.raw_log_dir / f"{stem}{suffix}.json"
            try:
                with path.open("x", encoding="utf-8") as file:
                    json.dump(record, file, ensure_ascii=False, indent=2)
                return str(path)
            except FileExistsError:
                occurrence += 1

    def _attempt(self, request: Request, validator: Callable, tag: str, key: str, *, retry_attempt=False):
        estimated_tokens = math.ceil(len(request.system + request.user) / 4)
        if self.max_request_tokens and estimated_tokens > self.max_request_tokens:
            raise LLMRequestError(f"Estimated input {estimated_tokens} exceeds max_request_tokens={self.max_request_tokens}; split the request")
        if self.consecutive_connection_errors >= 3:
            raise LLMNetworkError(NETWORK_BLOCKED)
        if self.call_budget is not None:
            self.call_budget.check()
        if self.call_budget and self.call_budget.wait_limit is not None:
            self.limiter.on_wait = lambda seconds: self.call_budget.wait_for_rate_limit(self.provider_name, self.model, seconds)
            self.limiter.token_history = self.call_budget.token_history.setdefault(self.provider_name, deque())
        self.limiter.wait(estimated_tokens)
        if self.call_budget is not None:
            self.call_budget.consume()
            self.call_budget.retries += int(retry_attempt)
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        response = None
        error_text = ""
        failed_text = ""
        connection_failure = False
        ok = False
        try:
            response = self._send(request)
            self.consecutive_connection_errors = 0
            if self.call_budget:
                self.call_budget.consecutive_connection_errors = 0
            self.last_raw_text = response.text
            result = validator(response.text)
            ok = True
            return result
        except Exception as error:
            error_text = self._safe_error(error)
            failed_text = getattr(error, "raw_text", "")
            if self.call_budget and self.call_budget.exempt_minute_limits and _status(error) == 429 and not _daily_quota(error):
                self.call_budget.used -= 1
                self.call_budget.retries -= int(retry_attempt)
            if response is None and _connection_failure(error):
                connection_failure = True
                self.connection_errors += 1
                self.consecutive_connection_errors += 1
                consecutive = self.consecutive_connection_errors
                if self.call_budget:
                    if self.provider_name != "gemini":
                        self.call_budget.used -= 1
                        self.call_budget.retries -= int(retry_attempt)
                    self.call_budget.connection_errors += 1
                    self.call_budget.consecutive_connection_errors += 1
                    consecutive = self.call_budget.consecutive_connection_errors
                if consecutive >= 3:
                    raise LLMNetworkError(NETWORK_BLOCKED) from None
            elif _status(error) is not None:
                self.consecutive_connection_errors = 0
                if self.call_budget:
                    self.call_budget.consecutive_connection_errors = 0
            raise
        finally:
            latency = round(time.perf_counter() - started, 6)
            in_tokens = response.in_tokens if response else None
            out_tokens = response.out_tokens if response else None
            if (self.call_budget and self.call_budget.wait_limit is not None and self.limiter.tokens_per_minute
                    and self.limiter.token_history and in_tokens is not None and out_tokens is not None):
                timestamp_start, _ = self.limiter.token_history[-1]
                self.limiter.token_history[-1] = (timestamp_start, in_tokens + out_tokens)
            record = {
                "provider": self.provider_name, "model": self.model,
                "timestamp": timestamp, "latency_s": latency,
                "in_tokens": in_tokens, "out_tokens": out_tokens,
                "temperature": request.temperature,
                "thinking_budget": getattr(self, "thinking_budget", None),
                "reasoning_effort": getattr(self, "reasoning_effort", None),
                "estimated_input_tokens": estimated_tokens,
                "prompt": {"system": request.system, "user": request.user},
                "schema": request.schema.model_json_schema() if request.schema else None,
                "structured": request.structured,
                "raw_response": response.text if response else failed_text,
                "ok": ok, "error": error_text,
                "connection_failure": connection_failure,
            }
            raw_path = self._raw_log(tag, key, record)
            self._usage(
                time=timestamp, tag=tag, provider=self.provider_name, model=self.model,
                cached=False, latency_s=latency, in_tokens=in_tokens,
                out_tokens=out_tokens, ok=ok, error=error_text,
            )
            self.last_call = {
                "cached": False, "latency_s": latency, "in_tokens": in_tokens,
                "out_tokens": out_tokens, "key": key, "raw_log": raw_path,
            }

    def _request(self, request: Request, validator: Callable, tag: str, key: str, *, repair_attempt=False):
        gemini = self.provider_name == "gemini"
        large_batch = gemini and tag.startswith("candidates_batch_") and len(tag.split("_E")) == 6
        attempt_limit = min(self.max_retries, 3) if gemini else self.max_retries
        attempts = 0
        rate_limited = False

        def attempt():
            nonlocal attempts
            attempts += 1
            return self._attempt(request, validator, tag, key, retry_attempt=repair_attempt or attempts > 1)

        def stop(state):
            return attempts >= attempt_limit or (
                large_batch and attempts >= 2 and _gemini_busy(state.outcome.exception()))

        def wait(state):
            nonlocal rate_limited
            rate_limited = _status(state.outcome.exception()) == 429
            if gemini and _gemini_busy(state.outcome.exception()):
                return 30 * 2 ** (attempts - 1)
            return _retry_wait(state)

        def sleep(seconds):
            if rate_limited and self.call_budget and self.call_budget.wait_limit is not None:
                self.call_budget.wait_for_rate_limit(self.provider_name, self.model, float(seconds))
            else:
                _retry_sleep(seconds)

        while True:
            retry = Retrying(
                stop=stop, wait=wait,
                sleep=sleep,
                retry=retry_if_exception(_transient), reraise=True,
            )
            try:
                return retry(attempt)
            except FormatRejected:
                if not request.structured:
                    raise
                if gemini and attempts >= attempt_limit:
                    raise LLMRequestError(f"[{tag}] Format fallback exhausted the {attempt_limit}-attempt limit") from None
                self.schema_fallback_used = True
                if request.schema is not None:
                    self._plain_schemas.add(request.schema)
                request = replace(request, structured=False)
            except (ValidationError, ValueError):
                raise
            except (LLMBudgetError, LLMNetworkError, LLMRateLimitError):
                raise
            except Exception as error:
                if _daily_quota(error):
                    self.quota_error = (
                        f"{self.provider_name} daily request quota exhausted. "
                        "Wait for the provider's reset or use an authorized key/model with available quota."
                    )
                    raise LLMQuotaError(f"[{tag}] {self.quota_error}") from None
                if large_batch and attempts >= 2 and _gemini_busy(error):
                    raise LLMBatchSplitError(f"[{tag}] {self._safe_error(error)}") from None
                raise LLMRequestError(f"[{tag}] {self._safe_error(error)}", status=_status(error),
                                      provider=self.provider_name, model=self.model) from None

    def _cached(
        self, key: str, schema: type[BaseModel] | None, tag: str,
        extra_validator: Callable[[BaseModel], list[str]] | None = None,
    ):
        if not self.use_cache:
            return None
        value = self._memory_cache.get(key)
        if value is None:
            try:
                entry = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
                value = schema.model_validate(entry["result"]) if schema else entry["result"]
                if schema is None and not isinstance(value, str):
                    return None
            except (OSError, ValueError, KeyError, TypeError):
                return None
        if schema is not None:
            try:
                # Revalidate memory values too, since callers can mutate models.
                schema.model_validate_json(value.model_dump_json())
                if extra_validator is not None and extra_validator(value):
                    return None
            except (ValueError, TypeError):
                return None
        self._memory_cache[key] = value
        self.last_call = {"cached": True, "latency_s": 0.0, "in_tokens": 0, "out_tokens": 0, "key": key}
        self._usage(
            time=datetime.now(timezone.utc).isoformat(), tag=tag,
            provider=self.provider_name, model=self.model, cached=True,
            latency_s=0.0, in_tokens=0, out_tokens=0, ok=True, error="",
        )
        return value

    def _store(self, key: str, result: BaseModel | str) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "result": result.model_dump(mode="json") if isinstance(result, BaseModel) else result,
            "in_tokens": self.last_call.get("in_tokens"),
            "out_tokens": self.last_call.get("out_tokens"),
        }
        (self.cache_dir / f"{key}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        self._memory_cache[key] = result

    def generate_json(
        self, system: str, user: str, schema: type[Model], tag: str,
        temperature: float | None = None,
        extra_validator: Callable[[BaseModel], list[str]] | None = None,
        max_repairs: int = 1,
    ) -> Model:
        if max_repairs not in (0, 1):
            raise ValueError("max_repairs must be 0 or 1")
        chosen_temperature = self.temperature if temperature is None else temperature
        key = self._key(system, user, schema, chosen_temperature)
        cached = self._cached(key, schema, tag, extra_validator)
        if cached is not None:
            return cached
        if self.quota_error:
            raise LLMQuotaError(f"[{tag}] {self.quota_error}")
        prompt = json_user_prompt(user, schema)
        last_error = ""

        def validate(raw: str) -> Model:
            result = schema.model_validate_json(_json_text(raw))
            if extra_validator is not None:
                errors = extra_validator(result)
                if errors:
                    raise ExtraValidationError(errors)
            return result

        for attempt in range(max_repairs + 1):
            repair = ""
            if attempt:
                repair = (
                    f"\n\nYour previous output failed validation: {last_error}. "
                    "Return ONLY valid JSON matching the schema."
                )
            request = Request(
                system, prompt + repair, chosen_temperature, schema,
                structured=schema not in self._plain_schemas,
            )
            try:
                result = self._request(
                    request, validate, tag, key, repair_attempt=bool(attempt),
                )
                self._store(key, result)
                return result
            except (ValidationError, ValueError) as error:
                if isinstance(error, ExtraValidationError):
                    messages = error.errors
                    hard_codes = {"unknown_shot_id", "unknown_line_id", "unknown_promo_id", "line_outside_clip", "shot_order", "invalid_json"}
                    repairable = bool(messages) and all(set(re.findall(r"\[([a-z_]+)\]", message)) & hard_codes for message in messages)
                elif isinstance(error, ValidationError):
                    messages = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors(include_url=False, include_input=False)]
                    repairable = True  # Invalid output syntax/schema is a hard protocol error.
                else:
                    messages = [str(error)]
                    repairable = False
                last_error = "; ".join(message[:200] for message in messages[:6])
                if repairable and attempt < max_repairs:
                    self.repair_events.append({
                        "tag": tag, "error_type": type(error).__name__,
                        "errors": getattr(error, "errors", []) if isinstance(error, ExtraValidationError) else [last_error],
                    })
                else:
                    break
        raise LLMOutputError(tag, self.last_raw_text, last_error)

    def generate_text(
        self, system: str, user: str, tag: str, temperature: float | None = None,
    ) -> str:
        chosen_temperature = self.temperature if temperature is None else temperature
        key = self._key(system, user, None, chosen_temperature)
        cached = self._cached(key, None, tag)
        if cached is not None:
            return cached
        if self.quota_error:
            raise LLMQuotaError(f"[{tag}] {self.quota_error}")
        request = Request(system, user, chosen_temperature)
        result = self._request(request, lambda raw: raw, tag, key)
        self._store(key, result)
        return result


class GeminiProvider(LLMClient):
    """Use google-genai; shared code handles retries and format fallback."""

    provider_name = "gemini"

    def __init__(self, *, api_key: str | None = None, client=None, timeout_seconds: float = 180, thinking_budget: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.timeout_seconds = timeout_seconds
        self.thinking_budget = thinking_budget
        self._api_key = api_key
        self._client = client
        self._secrets = [api_key] if api_key else []

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            key = self._api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise LLMRequestError("Set GEMINI_API_KEY in the environment or project .env")
            self._secrets = [key]
            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout_seconds * 1000), retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def _send(self, request: Request) -> Response:
        from google.genai import types

        config = {"temperature": request.temperature, "system_instruction": request.system}
        if self.thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        if request.schema is not None:
            config["response_mime_type"] = "application/json"
            if request.structured:
                config["response_schema"] = request.schema
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=request.user, config=types.GenerateContentConfig(**config),
            )
        except Exception as error:
            message = self._safe_error(error)
            if _status(error) == 404:
                try:
                    names = [model.name for model in self.client.models.list()]
                    available = ", ".join(names)
                except Exception as listing_error:
                    available = f"Could not list models: {self._safe_error(listing_error)}"
                print(f"Available Gemini models: {available}")
                raise LLMRequestError(f"Gemini model {self.model!r} not found. Available models: {available}") from None
            schema_rejected = (
                request.schema is not None and request.structured
                and (
                    isinstance(error, (ValueError, TypeError))
                    or (_status(error) == 400 and "schema" in message.lower())
                )
            )
            if schema_rejected:
                raise FormatRejected(message) from None
            raise
        usage = response.usage_metadata
        return Response(
            response.text or "",
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )


class OpenAICompatibleProvider(LLMClient):
    """Use Chat Completions for OpenAI, Groq, or local Ollama."""

    def __init__(
        self, *, provider_name: str = "openai", base_url: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY", client=None,
        api_key: str | None = None,
        reasoning_effort: str | None = None, max_completion_tokens: int | None = None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.provider_name = provider_name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._api_key = api_key
        self._secrets = [api_key] if api_key else []
        self._client = client
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = self._api_key or (os.getenv(self.api_key_env) if self.api_key_env else "ollama")
            if not key:
                raise LLMRequestError(f"Set {self.api_key_env} in the environment or project .env")
            self._secrets = [key] if self.api_key_env else []
            self._client = OpenAI(api_key=key, base_url=self.base_url, max_retries=0, timeout=60.0)
        return self._client

    def _send(self, request: Request) -> Response:
        options = {}
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens is not None:
            options["max_completion_tokens"] = self.max_completion_tokens
        if request.schema is not None and request.structured and request.schema.model_json_schema().get("type") != "array":
            options["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=request.temperature,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ], **options,
            )
        except Exception as error:
            message = self._safe_error(error)
            json_generation_failed = any(marker in message.lower() for marker in ("failed to validate json", "failed to generate json"))
            if "response_format" in options and _status(error) == 400 and json_generation_failed:
                body = getattr(error, "body", {})
                details = body.get("error", body) if isinstance(body, dict) else {}
                failed = details.get("failed_generation", "") if isinstance(details, dict) else ""
                raise FormatRejected(message, failed if isinstance(failed, str) else "") from None
            if "response_format" in options and _status(error) in {400, 422} and any(
                word in message.lower() for word in ("response_format", "json_object", "json mode")
            ):
                raise FormatRejected(message) from None
            raise
        usage = response.usage
        return Response(
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
        )


class MockProvider(LLMClient):
    """Return queued strings, response objects, or exceptions without networking."""

    provider_name = "mock"

    def __init__(self, responses=(), **kwargs):
        kwargs.setdefault("model", "mock")
        super().__init__(**kwargs)
        self.responses = deque(responses)
        self.calls: list[Request] = []

    def _send(self, request: Request) -> Response:
        self.calls.append(request)
        if not self.responses:
            raise LLMRequestError("Mock response queue is empty")
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, Response) else Response(response)


def get_llm(config: dict, provider: str | None = None, model: str | None = None) -> LLMClient:
    """Choose an adapter using the llm section and optional CLI overrides."""
    if not config.get("_interactive_run"):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = config.get("llm", config)
    chosen_provider = provider or settings.get("provider", "gemini")
    chosen_model = model or settings.get("model", "gemini-2.5-flash")
    provider_settings = dict(settings.get("providers", {}).get(chosen_provider, {}))
    shared = {
        "model": chosen_model, "temperature": settings.get("temperature", 0.4),
        "max_retries": settings.get("max_retries", 3),
        "requests_per_minute": settings.get("requests_per_minute", 8),
        "tokens_per_minute": provider_settings.pop("tokens_per_minute", settings.get("tokens_per_minute")),
        "max_request_tokens": request_token_limit(config, chosen_provider),
        "max_api_calls": settings.get("max_api_calls_per_run", 14),
        "use_cache": settings.get("use_cache", True),
        "cache_dir": settings.get("cache_dir", ".cache/llm"),
        "raw_log_dir": settings.get("raw_log_dir", "outputs/llm_raw"),
    }
    provider_settings.pop("max_request_tokens", None)
    if chosen_provider == "gemini":
        return GeminiProvider(api_key=config.get('_api_keys', {}).get('GEMINI_API_KEY'), timeout_seconds=provider_settings.get("timeout_seconds", 180), **shared)
    if chosen_provider == "mock":
        if config.get("_mock_upload"):
            from .mock_upload import UploadMockProvider
            return UploadMockProvider(config["parsed_dir"], **shared)
        return MockProvider(responses=settings.get("mock_responses", []), **shared)
    defaults = {
        "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
        "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
        "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
    }
    if chosen_provider not in defaults:
        raise ValueError(f"Unknown LLM provider: {chosen_provider!r}")
    options = defaults[chosen_provider] | provider_settings
    key_name = options.get('api_key_env')
    if key_name and config.get('_api_keys', {}).get(key_name):
        options['api_key'] = config['_api_keys'][key_name]
    return OpenAICompatibleProvider(provider_name=chosen_provider, **options, **shared)


class PingResult(BaseModel):
    ok: bool
    message: str


def demo_schema(episode: Episode) -> type[BaseModel]:
    """Validate three distinct, canonical shot IDs inside the JSON repair loop."""
    valid_ids = {shot.id for shot in episode.shots}

    class Moment(BaseModel):
        shot_id: str
        reason: str

        @field_validator("shot_id")
        @classmethod
        def known_shot(cls, value: str) -> str:
            if value not in valid_ids:
                raise ValueError(f"Unknown shot ID {value!r}; use a canonical ID from the episode")
            return value

    class DemoMoments(BaseModel):
        top_moments: list[Moment] = Field(min_length=3, max_length=3)

        @field_validator("top_moments")
        @classmethod
        def distinct_shots(cls, values: list[Moment]) -> list[Moment]:
            if len({value.shot_id for value in values}) != 3:
                raise ValueError("Choose three distinct shots")
            return values

    return DemoMoments


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--ping", action="store_true")
    action.add_argument("--demo-episode", type=int)
    cli.add_argument("--config", type=Path, default=Path("config.yaml"))
    cli.add_argument("--provider")
    cli.add_argument("--model")
    cli.add_argument("--max-api-calls", type=int)
    args = cli.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        client = get_llm(config, args.provider, args.model)
        if args.max_api_calls is not None:
            client.call_budget = CallBudget(args.max_api_calls)
        if args.ping:
            client.use_cache = False
            client.max_retries = 1
            result = client.generate_json(
                "You are a concise connectivity check. Return JSON.",
                'Return {"ok": true, "message": "<short>"}.', PingResult, "ping", max_repairs=0,
            )
        else:
            path = Path(config.get("parsed_dir", "parsed")) / f"episode_{args.demo_episode:03d}.json"
            episode = Episode.model_validate_json(path.read_text(encoding="utf-8"))
            result = client.generate_json(
                "Read the screenplay and select exactly three compelling promotional moments. Return JSON.",
                f"Use canonical IDs E{episode.number:03d}_S001 etc. The render's [S001] corresponds to that ID. "
                "Return top_moments with shot_id and a short reason for each.\n\n" + render_episode(episode),
                demo_schema(episode), f"demo_E{episode.number:03d}",
            )
        print(result.model_dump_json(indent=2))
        if not args.ping:
            by_id = {shot.id: shot for shot in episode.shots}
            for moment in result.top_moments:
                shot = by_id[moment.shot_id]
                print(f"{shot.id} | {format_timestamp(shot.effective_start)}-{format_timestamp(shot.effective_end)}")
                dialogue = [line for line in shot.lines if line.type == "dialogue"]
                for line in dialogue:
                    print(f'  {line.id} {line.speaker}: "{line.text}"')
                if not dialogue:
                    print("  (no dialogue in this shot)")
        info = client.last_call
        print(f"provider={client.provider_name} model={client.model} cached={info['cached']} "
              f"latency_s={info['latency_s']:.6f} tokens_in={info['in_tokens']} tokens_out={info['out_tokens']}")
        print(f"cache_key={info['key']} schema_fallback={client.schema_fallback_used}")
        print(f"api_calls={client.call_budget.used if client.call_budget else 'unlimited'} connection_errors={client.connection_errors}")
    except (LLMRequestError, LLMOutputError, OSError, ValueError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
