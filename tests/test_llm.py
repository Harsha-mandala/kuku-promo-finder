import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from src import llm
from src.llm import (
    GeminiProvider, LLMOutputError, LLMRequestError, MockProvider,
    OpenAICompatibleProvider, PingResult, RateLimiter, Response, demo_schema, get_llm,
)
from src.parser import parse_file


class Answer(BaseModel):
    value: int


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class APIError(Exception):
    def __init__(self, status_code, text="request failed"):
        self.status_code = status_code
        super().__init__(text)


def test_gemini_constructs_v1beta_sdk_client(settings, monkeypatch):
    from google import genai
    sdk = Mock()
    monkeypatch.setattr(genai, 'Client', sdk)
    client = GeminiProvider(model='test', api_key='fixture', timeout_seconds=45, **settings)
    assert client.client is sdk.return_value
    options = sdk.call_args.kwargs['http_options']
    assert options.api_version == 'v1beta'
    assert options.timeout == 45000


def test_openai_compatible_caps_completion_and_rejects_overflow(settings):
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='OK'))], usage=None)
    client = OpenAICompatibleProvider(provider_name='groq', model='test', client=sdk, request_tpm_limit=8000,
                                      max_completion_tokens=2048, **settings)
    assert client.generate_text('', 'a'*23000, 'fits') == 'OK'
    assert sdk.chat.completions.create.call_args.kwargs['max_completion_tokens'] == 1610
    with pytest.raises(LLMRequestError, match='too long for Groq'):
        client.generate_text('', 'a'*24000, 'overflow')
    assert sdk.chat.completions.create.call_count == 1


def test_gemini_transport_failures_count_and_backoff(tmp_path, monkeypatch):
    client = MockProvider(responses=[TimeoutError("timeout"), TimeoutError("timeout"), '{"value":1}'],
                          cache_dir=tmp_path/'cache', raw_log_dir=tmp_path/'raw', max_retries=3)
    client.provider_name = "gemini"
    client.call_budget = llm.CallBudget(3)
    client.limiter.wait = lambda *args: None
    waits = []
    monkeypatch.setattr(llm, '_retry_sleep', waits.append)
    client.generate_json('system', 'user', Answer, 'candidates_batch_E001_E002')
    assert client.call_budget.used == 3
    assert client.connection_errors == 2
    assert waits == [30, 60]


def test_gemini_thinking_is_sent_and_changes_cache_identity(settings):
    provider = GeminiProvider(model="gemini-2.5-flash", client=Mock(), thinking_budget=1024, **settings)
    provider.client.models.generate_content.return_value = Mock(text='ok', usage_metadata=None)
    provider._send(llm.Request('system', 'user', 0))
    config = provider.client.models.generate_content.call_args.kwargs['config']
    assert config.thinking_config.thinking_budget == 1024
    first = provider._key('system', 'user', None, 0)
    provider.thinking_budget = 512
    assert provider._key('system', 'user', None, 0) != first


def test_gemini_timeout_configuration(monkeypatch, settings):
    from google import genai
    factory = Mock()
    monkeypatch.setattr(genai, 'Client', factory)
    provider = get_llm({'llm': {'providers': {'gemini': {'timeout_seconds':180}}},
                        '_api_keys': {'GEMINI_API_KEY':'synthetic'}}, 'gemini')
    assert provider.client is factory.return_value
    assert factory.call_args.kwargs['http_options'].timeout == 180000


@pytest.fixture
def settings(tmp_path):
    return {
        "cache_dir": tmp_path / "cache",
        "raw_log_dir": tmp_path / "outputs" / "llm_raw",
        "requests_per_minute": 60,
    }


def mock_client(settings, responses, **options):
    client = MockProvider(responses=responses, **(settings | options))
    clock = FakeClock()
    client.limiter = RateLimiter(60, clock.clock, clock.sleep)
    return client


def usage_rows(client):
    with client.usage_log.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_cache_hit_returns_same_object_without_another_call(settings):
    client = mock_client(settings, ['{"value": 7}'])
    first = client.generate_json("system", "user", Answer, "first")
    second = client.generate_json("system", "user", Answer, "second")
    assert first is second
    assert len(client.calls) == 1
    assert client.last_call["cached"]
    assert [row["cached"] for row in usage_rows(client)] == ["False", "True"]
    assert usage_rows(client)[1]["in_tokens"] == "0"
    assert len(list(client.raw_log_dir.glob("*.json"))) == 1


def test_disk_cache_survives_new_client(settings):
    first = mock_client(settings, ['{"value": 7}'])
    first.generate_json("system", "user", Answer, "test")
    second = mock_client(settings, [])
    assert second.generate_json("system", "user", Answer, "test") == Answer(value=7)
    assert not second.calls


@pytest.mark.parametrize("raw", [
    '```json\n{"value": 7}\n```',
    '```json {"value": 7} ```',
    'Here is the answer: {"value": 7} Some trailing words.',
])
def test_json_cleanup_preserves_raw_log(settings, raw):
    client = mock_client(settings, [raw])
    assert client.generate_json("system", "user", Answer, "clean").value == 7
    record = json.loads(Path(client.last_call["raw_log"]).read_text(encoding="utf-8"))
    assert record["raw_response"] == raw


def test_one_repair_retry_and_separate_immutable_raw_logs(settings):
    client = mock_client(settings, ['{"value": "bad"}', '{"value": 8}'])
    assert client.generate_json("system", "user", Answer, "repair").value == 8
    assert len(client.calls) == 2
    assert "Your previous output failed validation:" in client.calls[1].user
    assert '{"value": "bad"}' not in client.calls[1].user
    assert 'Previous output:' not in client.calls[1].user
    logs = sorted(client.raw_log_dir.glob("*.json"))
    assert len(logs) == 2
    records = [json.loads(path.read_text(encoding="utf-8")) for path in logs]
    assert {record["raw_response"] for record in records} == {'{"value": "bad"}', '{"value": 8}'}
    assert [row["ok"] for row in usage_rows(client)] == ["False", "True"]


def test_invalid_output_exhaustion(settings):
    client = mock_client(settings, ["bad", "still bad", "last bad"], max_retries=3)
    with pytest.raises(LLMOutputError) as caught:
        client.generate_json("system", "user", Answer, "broken")
    assert len(client.calls) == 2
    assert caught.value.tag == "broken"
    assert caught.value.raw_text == "still bad"
    assert "still bad" in str(caught.value)
    assert len(usage_rows(client)) == 2
    assert not list(client.cache_dir.glob("*.json"))


def test_request_guard_rejects_before_provider_or_logs(settings):
    client = mock_client(settings, ['{"value": 7}'], max_request_tokens=5)
    with pytest.raises(LLMRequestError, match="max_request_tokens=5"):
        client.generate_json("system", "user", Answer, "too_large")
    assert not client.calls
    assert not list(client.raw_log_dir.glob("*.json"))


def test_validation_repair_budget_is_independent_of_transport_budget(settings):
    client = mock_client(settings, ["bad"] * 10, max_retries=10)
    with pytest.raises(LLMOutputError):
        client.generate_json("system", "user", Answer, "bounded")
    assert len(client.calls) == 2
    assert len(client.repair_events) == 1


def test_call_budget_counts_repairs_and_preserves_cache_hits(settings):
    client = mock_client(settings, ['{"value": 7}', 'bad', '{"value": 8}'], max_api_calls=2)
    assert client.generate_json("system", "first", Answer, "first").value == 7
    with pytest.raises(llm.LLMBudgetError):
        client.generate_json("system", "second", Answer, "repair_blocked")
    assert len(client.calls) == 2 and client.call_budget.used == 2
    assert client.generate_json("system", "first", Answer, "cached").value == 7
    assert len(client.calls) == 2
    assert len(list(client.raw_log_dir.glob("*.json"))) == 2


def test_call_budget_counts_failed_transport_attempts(settings, monkeypatch):
    monkeypatch.setattr(llm, "_retry_sleep", lambda delay: None)
    client = mock_client(settings, [APIError(500), '{"value": 7}'], max_api_calls=1)
    with pytest.raises(llm.LLMBudgetError):
        client.generate_json("system", "user", Answer, "retry_blocked")
    assert len(client.calls) == 1 and client.call_budget.used == 1


def test_call_budget_is_shared_across_clients(settings):
    first = mock_client(settings, ['{"value": 7}'])
    second = mock_client(settings, ['{"value": 8}'])
    first.call_budget = second.call_budget = llm.CallBudget(1)
    first.generate_json("system", "first", Answer, "one")
    with pytest.raises(llm.LLMBudgetError):
        second.generate_json("system", "second", Answer, "two")
    assert not second.calls


def test_three_no_response_failures_stop_without_consuming_budget(settings, monkeypatch):
    monkeypatch.setattr(llm, "_retry_sleep", lambda delay: None)
    client = mock_client(settings, [ConnectionError("network unreachable")] * 5,
                         max_api_calls=1, max_retries=10)
    with pytest.raises(llm.LLMNetworkError, match="network blocked — approve network or run locally"):
        client.generate_json("system", "user", Answer, "network")
    assert len(client.calls) == 3
    assert client.call_budget.used == 0 and client.call_budget.connection_errors == 3
    assert client.connection_errors == 3
    assert len(list(client.raw_log_dir.glob("*.json"))) == 3
    with pytest.raises(llm.LLMNetworkError):
        client.generate_json("system", "different", Answer, "later")
    assert len(client.calls) == 3


def test_response_resets_consecutive_connection_errors(settings, monkeypatch):
    monkeypatch.setattr(llm, "_retry_sleep", lambda delay: None)
    client = mock_client(settings, [ConnectionError(), '{"value": 7}', ConnectionError(), '{"value": 8}'],
                         max_api_calls=2, use_cache=False)
    assert client.generate_json("system", "one", Answer, "one").value == 7
    assert client.generate_json("system", "two", Answer, "two").value == 8
    assert client.call_budget.used == 2 and client.call_budget.connection_errors == 2
    assert client.call_budget.consecutive_connection_errors == 0


@pytest.mark.parametrize("provider,expected", [
    ("gemini", GeminiProvider), ("openai", OpenAICompatibleProvider),
    ("groq", OpenAICompatibleProvider), ("ollama", OpenAICompatibleProvider),
    ("mock", MockProvider),
])
def test_factory(provider, expected, settings, monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    client = get_llm({"llm": settings | {"provider": provider}}, model="selected-model")
    assert isinstance(client, expected)
    assert client.provider_name == provider
    assert client.model == "selected-model"
    if provider == "groq":
        assert client.base_url == "https://api.groq.com/openai/v1"
        assert client.api_key_env == "GROQ_API_KEY"
    if provider == "ollama":
        assert client.base_url == "http://localhost:11434/v1"
        assert client.api_key_env is None


def test_factory_overrides_and_bad_provider(settings):
    client = get_llm({"llm": settings | {"provider": "gemini", "model": "old"}}, "mock", "new")
    assert isinstance(client, MockProvider)
    assert client.model == "new"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm({}, "unknown")


def test_raw_log_and_usage_include_prompt_usage_and_latency(settings):
    client = mock_client(settings, [Response('{"value": 7}', 12, 4)])
    client.generate_json("system", "user", Answer, "../unsafe/tag")
    record = json.loads(Path(client.last_call["raw_log"]).read_text(encoding="utf-8"))
    assert Path(client.last_call["raw_log"]).parent == client.raw_log_dir
    assert record["provider"] == "mock"
    assert record["model"] == "mock"
    assert record["prompt"]["system"] == "system"
    assert record["prompt"]["user"].startswith("user")
    assert record["schema"]["title"] == "Answer"
    assert record["in_tokens"] == 12
    assert record["out_tokens"] == 4
    assert record["latency_s"] >= 0
    assert record["timestamp"].endswith("+00:00")
    row = usage_rows(client)[0]
    assert row["in_tokens"] == "12"
    assert row["out_tokens"] == "4"
    assert row["ok"] == "True"


def test_rate_limiter_uses_fake_time_for_two_calls(settings):
    client = MockProvider(responses=["first", "second"], use_cache=False, **settings)
    clock = FakeClock()
    client.limiter = RateLimiter(60, clock.clock, clock.sleep)
    assert client.generate_text("system", "one", "first") == "first"
    assert client.generate_text("system", "two", "second") == "second"
    assert clock.sleeps == [pytest.approx(1.0)]


def test_no_cache_and_temperature_override(settings):
    client = mock_client(settings, ["first", "second"], use_cache=False)
    client.generate_text("system", "user", "test", temperature=0)
    client.generate_text("system", "user", "test", temperature=0)
    assert len(client.calls) == 2
    assert client.calls[0].temperature == 0
    assert not client.cache_dir.exists()


def test_text_cache_and_key_separation(settings):
    client = mock_client(settings, ["text", '{"value": 2}', "different temp"])
    assert client.generate_text("system", "user", "test") == "text"
    assert client.generate_text("system", "user", "test") == "text"
    assert client.generate_json("system", "user", Answer, "test").value == 2
    assert client.generate_text("system", "user", "test", temperature=0) == "different temp"
    assert len(client.calls) == 3
    assert len(list(client.cache_dir.glob("*.json"))) == 3


def test_corrupt_cache_is_ignored(settings):
    client = mock_client(settings, ['{"value": 7}'])
    key = client._key("system", "user", Answer, 0.4)
    client.cache_dir.mkdir()
    (client.cache_dir / f"{key}.json").write_text("not JSON", encoding="utf-8")
    assert client.generate_json("system", "user", Answer, "test").value == 7
    assert len(client.calls) == 1


@pytest.mark.parametrize("failure", [APIError(429), APIError(503), TimeoutError("timed out")])
def test_transient_errors_retry_and_log_every_attempt(settings, monkeypatch, failure):
    clock = FakeClock()
    monkeypatch.setattr(llm.time, "sleep", clock.sleep)
    client = mock_client(settings, [failure, '{"value": 7}'])
    assert client.generate_json("system", "user", Answer, "network").value == 7
    assert len(client.calls) == 2
    assert len(usage_rows(client)) == 2
    assert usage_rows(client)[0]["ok"] == "False"
    assert clock.sleeps == [1]


def test_non_retryable_error_and_secrets_are_redacted(settings):
    client = mock_client(settings, [APIError(401, "invalid secret-value")])
    client._secrets = ["secret-value"]
    with pytest.raises(LLMRequestError) as caught:
        client.generate_text("system", "user", "auth")
    assert len(client.calls) == 1
    assert "secret-value" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "secret-value" not in client.usage_log.read_text(encoding="utf-8")
    assert "secret-value" not in Path(client.last_call["raw_log"]).read_text(encoding="utf-8")


def test_network_retry_exhaustion(settings, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda delay: None)
    client = mock_client(settings, [APIError(503), APIError(503), APIError(503)])
    with pytest.raises(LLMRequestError, match="network"):
        client.generate_text("system", "user", "network")
    assert len(client.calls) == 3


@pytest.mark.parametrize("failure", [
    APIError(400, "response_schema unsupported"),
    ValueError("additionalProperties is not supported in the Gemini API"),
])
def test_gemini_schema_rejection_falls_back_and_keeps_usage(settings, failure):
    sdk = SimpleNamespace(models=Mock())
    sdk.models.generate_content.side_effect = [
        failure,
        SimpleNamespace(text='{"value": 7}', usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3)),
    ]
    client = GeminiProvider(model="test", client=sdk, **settings)
    clock = FakeClock()
    client.limiter = RateLimiter(60, clock.clock, clock.sleep)
    assert client.generate_json("system", "user", Answer, "fallback").value == 7
    first, second = sdk.models.generate_content.call_args_list
    assert first.kwargs["config"].response_schema is Answer
    assert second.kwargs["config"].response_schema is None
    assert second.kwargs["config"].response_mime_type == "application/json"
    assert second.kwargs["config"].system_instruction == "system"
    assert '"title": "Answer"' in second.kwargs["contents"]
    assert client.schema_fallback_used
    assert len(usage_rows(client)) == 2
    assert client.last_call["in_tokens"] == 10


def test_gemini_missing_model_lists_names_and_stops(settings, capsys):
    sdk = SimpleNamespace(models=Mock())
    sdk.models.generate_content.side_effect = APIError(404)
    sdk.models.list.return_value = [SimpleNamespace(name="models/available-test-model")]
    client = GeminiProvider(model="missing", client=sdk, **settings)
    with pytest.raises(LLMRequestError, match="not found"):
        client.generate_text("system", "user", "missing")
    assert sdk.models.generate_content.call_count == 1
    assert "models/available-test-model" in capsys.readouterr().out


def test_compatible_json_mode_fallback(settings):
    create = Mock(side_effect=[
        APIError(400, "response_format json_object not supported"),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"value": 7}'))], usage=None),
    ])
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleProvider(model="local", client=sdk, **settings)
    clock = FakeClock()
    client.limiter = RateLimiter(60, clock.clock, clock.sleep)
    assert client.generate_json("system", "user", Answer, "fallback").value == 7
    assert create.call_args_list[0].kwargs["response_format"] == {"type": "json_object"}
    assert "response_format" not in create.call_args_list[1].kwargs
    assert client.last_call["in_tokens"] is None


def test_format_fallback_is_kept_during_json_repair(settings):
    sdk = SimpleNamespace(models=Mock())
    sdk.models.generate_content.side_effect = [
        APIError(400, "response_schema unsupported"),
        SimpleNamespace(text="invalid JSON", usage_metadata=None),
        SimpleNamespace(text='{"value": 7}', usage_metadata=None),
    ]
    client = GeminiProvider(model="test", client=sdk, **settings)
    clock = FakeClock()
    client.limiter = RateLimiter(60, clock.clock, clock.sleep)
    assert client.generate_json("system", "user", Answer, "repair").value == 7
    assert sdk.models.generate_content.call_count == 3
    assert sdk.models.generate_content.call_args_list[2].kwargs["config"].response_schema is None


def test_demo_requires_three_distinct_existing_shots(settings):
    episode = parse_file(Path(__file__).parent / "fixtures" / "sample_episode.txt")
    invalid = {"top_moments": [{"shot_id": "E099_S999", "reason": "bad"}] * 3}
    valid = {"top_moments": [{"shot_id": f"E099_S{index:03d}", "reason": "test"} for index in (1, 2, 3)]}
    client = mock_client(settings, [json.dumps(invalid), json.dumps(valid)])
    result = client.generate_json("system", "user", demo_schema(episode), "demo")
    assert len(client.calls) == 2
    assert result.top_moments[0].shot_id == "E099_S001"
    with pytest.raises(ValueError):
        demo_schema(episode).model_validate({"top_moments": valid["top_moments"][:2]})
    with pytest.raises(ValueError):
        demo_schema(episode).model_validate({"top_moments": [valid["top_moments"][0]] * 3})


def test_extra_validator_repairs_once_and_caches_only_valid_result(settings):
    client = mock_client(settings, ['{"value": 1}', '{"value": 7}'])
    validator = lambda result: ["[unknown_line_id] value must be at least 5"] if result.value < 5 else []
    result = client.generate_json("system", "user", Answer, "domain", extra_validator=validator)
    assert result.value == 7
    assert len(client.calls) == 2
    assert "value must be at least 5" in client.calls[1].user
    assert [row["ok"] for row in usage_rows(client)] == ["False", "True"]
    assert len(client.repair_events) == 1
    entry = json.loads(next(client.cache_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert entry["result"] == {"value": 7}


def test_soft_extra_validator_does_not_trigger_repair(settings):
    client = mock_client(settings, ['{"value": 1}'] * 3)
    with pytest.raises(LLMOutputError, match="domain rejection"):
        client.generate_json(
            "system", "user", Answer, "domain", extra_validator=lambda result: ["domain rejection"],
        )
    assert len(client.calls) == 1
    assert not list(client.cache_dir.glob("*.json"))
    assert all(row["ok"] == "False" for row in usage_rows(client))


def test_cache_hits_must_pass_current_extra_validator(settings):
    client = mock_client(settings, ['{"value": 1}', '{"value": 7}'])
    client.generate_json("system", "user", Answer, "domain")
    result = client.generate_json(
        "system", "user", Answer, "domain",
        extra_validator=lambda result: ["must be 7"] if result.value != 7 else [],
    )
    assert result.value == 7
    assert len(client.calls) == 2
    assert not client.last_call["cached"]


def test_daily_quota_stops_retries_and_new_calls_but_keeps_cache_usable(settings):
    client = mock_client(settings, ['{"value": 7}', APIError(429, "Quota exceeded: generate requests per day limit")])
    client.generate_json("system", "cached", Answer, "saved")
    with pytest.raises(LLMRequestError, match="daily request quota"):
        client.generate_json("system", "uncached", Answer, "blocked")
    assert len(client.calls) == 2
    with pytest.raises(LLMRequestError, match="daily request quota"):
        client.generate_text("system", "another", "blocked")
    assert len(client.calls) == 2
    assert client.generate_json("system", "cached", Answer, "saved").value == 7
    assert client.last_call["cached"]


def test_token_throttle_rolling_minute_and_single_request_limit():
    clock = FakeClock()
    limiter = RateLimiter(60,clock.clock,clock.sleep,tokens_per_minute=100)
    limiter.wait(70)
    limiter.wait(40)
    assert clock.now == pytest.approx(60)
    assert max(clock.sleeps) <= 30
    with pytest.raises(LLMRequestError,match="exceeds tokens_per_minute"):
        limiter.wait(101)


def test_per_minute_429_honors_retry_after(settings,monkeypatch):
    clock=FakeClock()
    monkeypatch.setattr(llm.time,"sleep",clock.sleep)
    failure=APIError(429,"tokens per minute limit")
    failure.response=SimpleNamespace(headers={"retry-after":"12.5"})
    client=mock_client(settings,[failure,'{"value":7}'])
    assert client.generate_json("system","user",Answer,"minute").value==7
    assert clock.sleeps==[12.5]
    assert len(client.calls)==2


@pytest.mark.parametrize('message',['Failed to validate JSON. See failed_generation.','Failed to generate JSON. See failed_generation.'])
def test_provider_json_validation_failure_preserves_raw_and_falls_back(settings,message):
    failure=APIError(400,message)
    failure.body={'failed_generation':'{"value": invalid}'}
    create=Mock(side_effect=[failure,SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":7}'))],usage=None)])
    sdk=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client=OpenAICompatibleProvider(model='test',client=sdk,**settings)
    clock=FakeClock()
    client.limiter=RateLimiter(60,clock.clock,clock.sleep)
    assert client.generate_json('system','user',Answer,'native_json').value==7
    assert 'response_format' not in create.call_args_list[1].kwargs
    records=[json.loads(path.read_text(encoding='utf-8')) for path in client.raw_log_dir.glob('*.json')]
    assert any(record['raw_response']=='{"value": invalid}' for record in records)


def test_compatible_generation_limits_are_forwarded(settings):
    create=Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":7}'))],usage=None))
    sdk=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client=OpenAICompatibleProvider(model='test',client=sdk,reasoning_effort='low',max_completion_tokens=4096,**settings)
    assert client.generate_json('system','user',Answer,'limits').value==7
    assert create.call_args.kwargs['reasoning_effort']=='low'
    assert create.call_args.kwargs['max_completion_tokens']==4096


def test_gemini_format_fallback_does_not_reset_attempt_limit(tmp_path, monkeypatch):
    client = MockProvider(responses=[llm.FormatRejected('schema'), APIError(503, 'overloaded'), APIError(503, 'overloaded'), '{"value":1}'],
                          cache_dir=tmp_path/'cache', raw_log_dir=tmp_path/'raw', max_retries=9)
    client.provider_name = 'gemini'
    client.call_budget = llm.CallBudget(9)
    client.limiter.wait = lambda *args: None
    monkeypatch.setattr(llm, '_retry_sleep', lambda *args: None)
    with pytest.raises(LLMRequestError):
        client.generate_json('system', 'user', Answer, 'candidates_batch_E001_E002')
    assert client.call_budget.used == 3
    assert len(client.calls) == 3
