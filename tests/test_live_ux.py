from pathlib import Path

import pytest

from src import candidates, llm, pipeline
from src.run_status import UNKNOWN_ERROR, describe_failure, error_details
from src.ui.data import DEMO_ROOT, PROJECT_ROOT, upload_config
from src.ui.session import upload_run_fingerprint
from test_judge_rank_pipeline import setup_pipeline
from test_llm import APIError, mock_client, settings  # pytest fixture


@pytest.mark.parametrize(("error", "kind", "guidance"), [
    (llm.LLMQuotaError("daily"), "Daily quota", "quota resets"),
    (llm.LLMRequestError("/tmp/private", status=429), "Per-minute rate limit (429)", "about one minute"),
    (llm.LLMRequestError("/tmp/private", status=503), "Provider busy (503)", "switch providers"),
    (llm.LLMRequestError("/tmp/private", status=504), "Provider busy (504)", "switch providers"),
    (llm.LLMRequestError("/tmp/private", status=401), "Authentication (401)", "valid key"),
    (llm.LLMRequestError("/tmp/private", status=403), "Authentication (403)", "valid key"),
    (llm.LLMBudgetError("spent"), "Call budget", "increase max api calls"),
])
def test_interactive_failure_stops_before_validation(tmp_path, monkeypatch, error, kind, guidance):
    source, config = setup_pipeline(tmp_path)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(pipeline, 'run_candidates', fail)
    monkeypatch.setattr(pipeline, 'run_validation', lambda *args, **kwargs: pytest.fail('Validation ran'))
    result = pipeline.run_pipeline(config, source, 'test', {'interactive': True})
    failure = result['failure']
    assert failure['stage'] == 'candidates' and failure['kind'] == kind
    assert guidance in failure['next_step'].lower()
    assert failure['done'] == [] and failure['pending'] == [99]
    assert '/tmp' not in str(failure)


def test_missing_editor_key_blocks_before_any_client(tmp_path, monkeypatch):
    source, config = setup_pipeline(tmp_path)
    config['models']['editor'] = {'provider': 'openai', 'model': 'test'}
    config['editor'] = {'enabled': True}
    monkeypatch.setattr(candidates, 'get_llm', lambda *args: pytest.fail('Client created'))
    result = pipeline.run_pipeline(config, source, 'test', {'interactive': True})
    assert result['failure']['stage'] == 'editor'
    assert result['failure']['kind'] == 'Missing API key'
    assert result['failure']['message'] == 'OPENAI_API_KEY is missing.'


@pytest.mark.parametrize(('status','message','hint'), [
    (400, 'Developer instruction is not enabled. Use v1beta.', 'version mismatch'),
    (400, 'API_KEY_INVALID', 'not enabled for Gemini API'),
    (403, 'PERMISSION_DENIED', 'not enabled for Gemini API'),
    (429, 'RESOURCE_EXHAUSTED', 'NEW project'),
    (413, 'TPM limit exceeded', 'one episode per call'),
    (503, 'Unavailable', 'press Resume'),
    (504, 'Deadline exceeded', 'press Resume'),
])
def test_provider_error_hints_and_masked_details(status, message, hint):
    error = llm.LLMRequestError(message, status=status, error_class='ClientError',
                                api_message=f'{message} key=fixture-key')
    assert hint in describe_failure(error)[2]
    assert error_details(error, ['fixture-key']) == ('ClientError', status, f'{message} key=[REDACTED]')
    assert error_details(llm.LLMRequestError(''))[2] == UNKNOWN_ERROR


def test_hosted_default_is_30_in_dry_run_and_sidebar(tmp_path, monkeypatch, capsys):
    from streamlit.testing.v1 import AppTest
    source, config = setup_pipeline(tmp_path)
    config['llm'] = {'max_api_calls_per_run': 14}
    pipeline.run_pipeline(config, source, 'test', {'hosted_keys': True, 'dry_run': True})
    assert 'Shared call budget=30' in capsys.readouterr().out
    monkeypatch.setenv('PUBLIC_DEPLOY','1')
    monkeypatch.setenv('GEMINI_API_KEY','fixture-gemini')
    monkeypatch.setenv('GROQ_API_KEY','fixture-groq')
    app = AppTest.from_file(str(PROJECT_ROOT/'app.py'), default_timeout=30).run()
    app.radio[0].set_value('Run on my files').run()
    assert next(x for x in app.number_input if x.label=='Max API calls').value == 30
    assert any('20 episodes and 30 API calls' in x.value for x in app.caption)


def test_key_probe_never_retries_or_exposes_secret(monkeypatch):
    from types import SimpleNamespace
    from src.ui import appview
    sent = []
    def fake(config, provider, model):
        assert config['llm']['max_retries'] == 1
        def respond(*args):
            sent.append(provider)
            raise llm.LLMRequestError('API_KEY_INVALID key=fixture-secret', status=400)
        return SimpleNamespace(generate_text=respond)
    monkeypatch.setattr(appview, 'get_llm', fake)
    result = list(appview.test_selected_keys({}, {'GEMINI_API_KEY':'fixture-secret'}, {'gemini':'fixture','groq':'fixture'}))
    assert sent == ['gemini']
    assert result[0] == ('gemini', False, 'Key is wrong or not enabled for Gemini API.')
    assert result[1][1] is False and 'missing' in result[1][2]


def test_bad_key_panel_renders_masked_provider_message(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('PUBLIC_DEPLOY','1')
    monkeypatch.setenv('GEMINI_API_KEY','fixture-gemini')
    monkeypatch.setenv('GROQ_API_KEY','fixture-groq')
    def fail(*args, **kwargs):
        error = llm.LLMRequestError('API_KEY_INVALID', status=400, provider='gemini', model='gemini-2.5-flash',
                                    error_class='ClientError', api_message='API_KEY_INVALID: key=fixture-gemini')
        error.episodes = [1]
        raise error
    monkeypatch.setattr(pipeline, 'run_candidates', fail)
    app = AppTest.from_file(str(PROJECT_ROOT/'app.py'), default_timeout=30).run()
    app.radio[0].set_value('Run on my files').run()
    episode = DEMO_ROOT/'episodes/episode_001.txt'
    next(x for x in app.file_uploader if x.label.startswith('Input pack')).set_value(
        [(episode.name, episode.read_bytes(), 'text/plain')]).run()
    next(x for x in app.button if x.label=='Find promos').click().run()
    assert not app.exception
    assert len(app.status) == 1 and app.status[0].proto.expanded
    detail = next(x.value for x in app.text if 'Episode 1 ·' in x.value)
    assert 'gemini / gemini-2.5-flash · ClientError · HTTP 400' in detail
    assert 'API_KEY_INVALID: key=[REDACTED]' in detail
    assert 'fixture-gemini' not in detail
    assert any('Key is wrong or not enabled for Gemini API' in x.value for x in app.markdown)
    print(app.status[0].label, detail, 'Key is wrong or not enabled for Gemini API.', sep='\n')


def test_minute_429_exempt_from_hosted_budget_and_wait_is_reported(settings, monkeypatch):
    monkeypatch.setattr(llm, '_retry_sleep', lambda seconds: None)
    notices = []
    client = mock_client(settings, [APIError(429, 'try again in 9 seconds'), 'ok'])
    client.limiter.wait = lambda tokens: None
    client.call_budget = llm.CallBudget(1, exempt_minute_limits=True, wait_limit=180,
                                       progress=lambda *args: notices.append(args))
    assert client.generate_text('system', 'user', 'tag') == 'ok'
    assert client.call_budget.used == 1 and client.call_budget.retries == 1
    assert client.call_budget.waited == 9 and [seconds for _, seconds in notices] == [9, 4]
    blocked = mock_client(settings, [APIError(429, 'try again in 181 seconds')])
    blocked.call_budget = llm.CallBudget(1, exempt_minute_limits=True, wait_limit=180)
    with pytest.raises(llm.LLMRateLimitError):
        blocked.generate_text('system', 'other user', 'tag')
    assert blocked.call_budget.used == 0


def test_fingerprint_covers_file_model_and_prompt_versions(tmp_path):
    prompt = tmp_path / 'judge.md'
    prompt.write_text('v1')
    files = {'one.txt': b'one'}
    models = {'judge': ('groq', 'm1')}
    one = upload_run_fingerprint(files, models, 'all', tmp_path)
    assert one != upload_run_fingerprint(files, {'judge': ('groq', 'm2')}, 'all', tmp_path)
    assert one != upload_run_fingerprint({'one.txt': b'two'}, models, 'all', tmp_path)
    prompt.write_text('v2')
    assert one != upload_run_fingerprint(files, models, 'all', tmp_path)


def test_interactive_budget_stop_resumes_remaining_episodes(tmp_path):
    import yaml
    config = upload_config(yaml.safe_load((PROJECT_ROOT/'config.yaml').read_text(encoding='utf-8')),
                           tmp_path, {}, mock=True)
    source = DEMO_ROOT/'episodes'
    stopped = pipeline.run_pipeline(config, source, 'mock_resume', {'interactive': True, 'max_api_calls': 2},
                                    batch_size=1)
    assert stopped['failure']['kind'] == 'Call budget'
    assert stopped['failure']['done'] == [1]
    assert stopped['failure']['pending'] == [2, 3, 4]
    saved = Path(config['output_dir'])/'candidates_mock_resume/episode_001.json'
    before = saved.read_bytes()
    resumed = pipeline.run_pipeline(config, source, 'mock_resume', {'interactive': True, 'max_api_calls': 30},
                                    batch_size=1)
    assert not resumed['stopped']
    assert resumed['candidate_stats']['episodes_skipped'] == [1]
    assert saved.read_bytes() == before


def test_public_provider_list_and_model_switch_isolates_runs(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.ui import appview

    monkeypatch.setenv('PUBLIC_DEPLOY', '1')
    for name in ('OPENAI_API_KEY','GEMINI_API_KEY','GROQ_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    paths = []
    def fake(config, input_dir, tag, overrides, progress):
        paths.append(Path(config['output_dir']))
        return {'stopped': None}
    monkeypatch.setattr(appview, 'run_pipeline', fake)
    app = AppTest.from_file(str(PROJECT_ROOT/'app.py'), default_timeout=30).run()
    app.radio[0].set_value('Run on my files').run()
    candidate = next(item for item in app.selectbox if item.label == 'Candidate provider')
    assert all('ollama' not in option for option in candidate.options)
    assert 'openai — needs your API key' in candidate.options
    file = (Path(__file__).parent/'fixtures/sample_episode.txt').read_bytes()
    next(item for item in app.file_uploader if item.label.startswith('Input pack')).set_value(
        [('episode.txt',file,'text/plain')]).run()
    def find():
        next(item for item in app.button if item.label == 'Find promos').click().run()
        assert not app.exception
    find()
    find()
    next(item for item in app.text_input if item.label == 'Edit candidate model').set_value('different-model').run()
    find()
    assert paths[0] == paths[1] and paths[2] != paths[0]
    assert paths[0].parent.exists()
