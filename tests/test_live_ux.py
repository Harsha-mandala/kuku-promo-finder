from pathlib import Path

import pytest

from src import candidates, llm, pipeline
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
