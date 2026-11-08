import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ui.data import PROJECT_ROOT, load_demo, private_config
from src.ui.helpers import decisions_csv, filter_promos, redact, segment_text
from src.ui.session import clear_workspace, new_workspace, resolve_credentials, unpack_uploads, write_uploads
from src.ui.theme import card_html
from src.judge import JudgeBatch, JudgeScore, SavedJudgeScore, run_judge, verify_safety
from src.rank import rank_promos
from src.pipeline import run_pipeline
from test_judge_rank_pipeline import setup_judge, mock, score, setup_pipeline
from test_validate import timeline, proposal
from src.validate import validate_candidate


def test_quality_tab_labels_demo_and_shows_saved_tables():
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert any('Demo data' in info.value for info in app.info)
    assert any('System comparison' in item.value for item in app.markdown)
    assert any('Judge versus human' in item.value for item in app.markdown)
    assert any('Cost' in item.value for item in app.markdown)


def test_quality_primary_table_serializes_without_arrow_fallback():
    import pyarrow as pa
    import pandas as pd
    from src.eval import primary_comparison_rows
    metrics=json.loads((PROJECT_ROOT/'demo/precomputed/results/eval_demo/metrics.json').read_text(encoding='utf-8'))
    table=pa.Table.from_pandas(pd.DataFrame(primary_comparison_rows(metrics)),preserve_index=False)
    assert table.num_rows==4


def test_public_deploy_hides_local_runs(monkeypatch):
    from src.ui.data import local_runs_allowed
    headers={'Host':'localhost:8501'}
    assert not local_runs_allowed({'PUBLIC_DEPLOY':'1'}, headers)
    assert not local_runs_allowed({}, {'Host':'app.streamlit.app'})
    assert not local_runs_allowed({}, {**headers,'X-Forwarded-Host':'app.streamlit.app'})
    monkeypatch.setenv('PUBLIC_DEPLOY','1')
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert 'Open existing run' not in app.radio[0].options


def test_mock_upload_runs_all_stages_on_demo_episodes(tmp_path):
    import yaml
    from src.ui.data import DEMO_ROOT, load_bundle, upload_config
    base=yaml.safe_load((PROJECT_ROOT/'config.yaml').read_text(encoding='utf-8'))
    files=unpack_uploads([(path.name,path.read_bytes()) for path in (DEMO_ROOT/'episodes').glob('*.txt')])
    input_dir=write_uploads(tmp_path,files)
    config=upload_config(base,tmp_path,{},mock=True)
    result=run_pipeline(config,input_dir,'session_mock',max_api_calls=20)
    assert not result['stopped'], result.get('error')
    bundle=load_bundle(tmp_path/'parsed',tmp_path/'outputs','session_mock')
    assert len(bundle['episodes'])==4
    assert bundle['ranked'] and bundle['trailers'] and bundle['editor_order']
    with (tmp_path/'outputs/llm_usage.csv').open(encoding='utf-8',newline='') as stream:
        assert {row['provider'] for row in csv.DictReader(stream)}=={'mock'}


def test_download_pack_contains_every_segment_and_requested_file():
    from src.ui.downloads import download_files, download_zip, posting_rows
    bundle=load_demo()
    files=download_files(bundle,{})
    rows=list(csv.DictReader(io.StringIO(files['editor_ready.csv'].decode())))
    posting=posting_rows(bundle)
    assert len(rows)==sum(len(row['segments']) for row in posting)
    assert [int(row['final_rank']) for row in rows]==sorted(int(row['final_rank']) for row in rows)
    assert [row['segment_no'] for row in rows if row['type']=='trailer']==['1','2']
    with zipfile.ZipFile(io.BytesIO(download_zip(files))) as archive:
        assert set(archive.namelist())=={
            'promos_ranked.csv','promos_ranked.json','trailers_ranked.csv','trailers_ranked.json',
            'promos_all.csv','report.md','editor_ready.csv','editor_decisions.csv','story_bible.json','eval_report.md'}
        assert archive.read('editor_ready.csv')==files['editor_ready.csv']


def test_in_app_guide_has_all_nine_sections():
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    labels={item.label for item in app.expander}
    assert {'1. Overview','2. Modes','3. Tabs and controls','4. Glossary and scoring','5. Run a new show',
            '6. Downloads','7. Quality and evaluation','8. Limits and privacy','9. FAQ'}.issubset(labels)


def test_credential_resolution_includes_editorial_stage_providers(monkeypatch):
    from src.ui import appview
    requested=[]
    monkeypatch.setattr(appview,'model_controls',lambda *args:('groq','mock'))
    def resolve(pasted,secrets,environ,providers):
        requested.append(providers)
        return {},False
    monkeypatch.setattr(appview,'resolve_credentials',resolve)
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=30).run()
    app.radio[0].set_value('Run on my files').run()
    assert not app.exception
    assert requested[-1]=={'gemini','groq'}


def test_demo_loads_original_four_episode_bundle():
    bundle=load_demo()
    assert len(bundle['episodes'])==4 and len(bundle['ranked'])==11
    assert all(90<=ep.duration<=120 for ep in bundle['episodes'])
    assert min(row['duration_s'] for row in bundle['ranked'])==22
    assert max(row['duration_s'] for row in bundle['ranked'])==48
    assert bundle['bible']['title_guess']=='The Midnight Patent'
    assert all(row['post_priority'] is not None for row in bundle['ranked'])
    assert any(not row['ad_safety_verified'] for row in bundle['ranked'])
    assert all(ep.source_file==f'episode_{ep.number:03d}.txt' for ep in bundle['episodes'])


def test_format_filters_and_safe_editor_export():
    rows=load_demo()['ranked']
    subset=filter_promos(rows,(1,2),0,True,['ok'])
    assert subset and all(r['episode']<=2 and r['ad_safety']=='safe' for r in subset)
    assert filter_promos(rows,(1,4),0,False,[])==[]
    assert segment_text(rows[0]).startswith('E') and '–' in segment_text(rows[0])
    data=decisions_csv({'TEST':{'decision':'approve','note':'A comma, and\na second line'}}).decode()
    assert list(csv.DictReader(io.StringIO(data)))[0]['note']=='A comma, and\na second line'
    assert "'=SUM" in decisions_csv({'X':{'decision':'reject','note':'=SUM(A1)'}}).decode()


def test_cards_escape_input_and_show_unverified():
    row=load_demo()['ranked'][0]|{'hook_text':'<script>bad()</script>','ad_safety_verified':False}
    html=card_html(row)
    assert '<script>' not in html and '&lt;script&gt;' in html and 'unverified' in html


def test_ui_html_helpers_remove_blank_lines_and_indentation():
    from src.ui.helpers import render_html, safety_chip, score_bars
    demo = load_demo()
    assert render_html("\n    <div>\n\n        text\n    </div>\n") == "<div> text </div>"
    fragments = [fragment for row in demo['ranked'] + demo['trailers']
                 for fragment in (card_html(row), safety_chip(row), score_bars(row))]
    assert all(line.strip() and not line.startswith('    ') for body in fragments for line in body.splitlines())


def test_demo_html_fragments_use_html_elements_without_indented_lines():
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    fragments = [item.proto.body for item in app.get('html')]
    assert any('Trailer ·' in body and 'score-track' in body for body in fragments)
    assert any('timeline-track' in body for body in fragments)
    assert all(line.strip() and not line.startswith('    ') for body in fragments for line in body.splitlines())
    assert all('<div class=' not in item.value for item in [*app.markdown, *app.get('code')])


def test_own_keys_override_secrets_without_environment_mutation():
    env={'GROQ_API_KEY':'environment'}
    keys,hosted=resolve_credentials({'GROQ_API_KEY':'own'},{'GROQ_API_KEY':'hosted'},env,{'groq'})
    assert keys=={'GROQ_API_KEY':'own'} and not hosted and env=={'GROQ_API_KEY':'environment'}
    keys,hosted=resolve_credentials({}, {'GROQ_API_KEY':'hosted'},env,{'groq'})
    assert hosted and keys['GROQ_API_KEY']=='hosted'
    assert redact('failure own',{'key':'own'})=='failure [redacted]'


def test_session_workspace_is_private_and_clear_deletes_it():
    workspace=new_workspace()
    root=Path(workspace.name)
    try:
        write_uploads(root,{'episode.txt':b'Synthetic'})
        config=private_config({'llm':{}},root,{'GROQ_API_KEY':'not-a-real-key'})
        assert config['_api_keys']['GROQ_API_KEY']=='not-a-real-key'
        assert Path(config['output_dir']).is_relative_to(root)
        assert not list(root.glob('*.yaml')) and not (root/'.env').exists()
    finally:
        clear_workspace(workspace)
    assert not root.exists()


@pytest.mark.parametrize('name',['../episode.txt','folder/../../episode.txt','C:/episode.txt','..\\episode.txt'])
def test_zip_traversal_rejected(name):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr(name,'Synthetic')
    with pytest.raises(ValueError,match='unsafe path'):
        unpack_uploads([('pack.zip',data.getvalue())])


def test_zip_nested_txt_and_duplicate_detection():
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr('folder/episode.txt','Synthetic')
        archive.writestr('README.md','Ignore')
    assert unpack_uploads([('pack.zip',data.getvalue())])=={'episode.txt':b'Synthetic'}
    with pytest.raises(ValueError,match='Duplicate'):
        unpack_uploads([('episode.txt',b'A'),('EPISODE.txt',b'B')])


def test_hosted_episode_limit_stops_before_api_client(tmp_path,monkeypatch):
    source,config=setup_pipeline(tmp_path)
    template=(source/'episode.txt').read_text(encoding='utf-8')
    (source/'episode.txt').unlink()
    for number in range(1,22):
        (source/f'{number}.txt').write_text(template.replace('Episode 99',f'Episode {number}'),encoding='utf-8')
    from src import candidates
    monkeypatch.setattr(candidates,'get_llm',lambda *a:pytest.fail('Hosted guard must precede client creation'))
    with pytest.raises(ValueError,match='at most 20'):
        run_pipeline(config,source,'hosted',{'hosted_keys':True,'dry_run':True,'max_api_calls':100})


def test_shared_pipeline_progress_and_hosted_call_cap(tmp_path,monkeypatch):
    from src import candidates,judge
    from src.models import CandidateList
    from test_candidates import proposal as candidate_proposal
    from test_context import synthetic_bible
    source,config=setup_pipeline(tmp_path)
    candidate=candidate_proposal(start_shot_id='S001',end_shot_id='S005',hook_line_id='L002',cliffhanger_line_id='L010')
    generator=mock(tmp_path/'gen',[synthetic_bible().model_dump_json(),json.dumps([CandidateList(episode=99,candidates=[candidate]).model_dump()])])
    reviewer=mock(tmp_path/'judge',[JudgeBatch([score()]).model_dump_json()])
    monkeypatch.setattr(candidates,'get_llm',lambda *a:generator)
    monkeypatch.setattr(judge,'get_llm',lambda *a:reviewer)
    events=[]
    result=run_pipeline(config,source,'test',{'hosted_keys':True,'max_api_calls':100},events.append)
    assert result['candidate_stats']['api_budget_limit']==14
    stages=[event['stage'] for event in events if event['state']=='running']
    assert stages==['parse','bible','candidates','validate','judge','rank']
    assert any(event.get('episode')==99 and event['stage']=='judge' for event in events)


def test_in_memory_keys_are_not_put_in_environment_or_raw_errors(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src import llm
    monkeypatch.setattr(llm,'load_dotenv',lambda *a,**kw:None)
    key='SYNTHETIC-KEY-ONLY-FOR-TESTS'
    previous=os.environ.get('GROQ_API_KEY')
    config={'_api_keys':{'GROQ_API_KEY':key},'llm':{'provider':'groq','max_retries':1,
        'cache_dir':str(tmp_path/'cache'),'raw_log_dir':str(tmp_path/'raw')}}
    client=llm.get_llm(config,model='test-model')
    create=Mock(side_effect=RuntimeError('failure '+key))
    client._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(llm.LLMRequestError):
        client.generate_text('system','user','synthetic_secret')
    assert os.environ.get('GROQ_API_KEY')==previous
    assert all(key not in p.read_text() for p in tmp_path.rglob('*') if p.is_file())


def test_public_manifest_never_includes_private_folders():
    from tools.public_manifest import public_files
    files=public_files()
    assert 'demo/episodes/episode_001.txt' in files
    assert not any(name.startswith(('data/','parsed/','outputs/','.cache/','.venv/')) or name=='.env' for name in files)


def test_post_priority_weights_and_legacy_fallback():
    p=validate_candidate(timeline(),proposal())
    full=score(post_priority=10)
    r=rank_promos([p],[full],{})['ranked'][0]
    expected=.25*.75+.25*.5+.20*.75+.10*1+.20*1
    assert r['judge_part']==pytest.approx(expected)
    old=SavedJudgeScore.model_validate(full.model_dump(exclude={'post_priority'}))
    r=rank_promos([p],[old],{})['ranked'][0]
    assert r['judge_part']==pytest.approx(.30*.75+.30*.5+.25*.75+.15*1)
    assert r['formula_note'].startswith('old formula') and r['post_priority'] is None


def test_mock_judge_keeps_uncited_caution_without_repair(tmp_path):
    config,ep,promos=setup_judge(tmp_path)
    first=score(promos[0].id,ad_safety='caution',ad_safety_reason='',post_priority=9)
    second=score(promos[1].id,ad_safety='caution',ad_safety_reason='Threat at E099_L006.',post_priority=7)
    client=mock(tmp_path,[JudgeBatch([first]).model_dump_json(),JudgeBatch([second]).model_dump_json()])
    stats=run_judge(config,'test',client=client)
    saved=JudgeBatch.model_validate_json(Path(config['output_dir'],'judged_test/scores.json').read_text()).root
    assert len(client.calls)==2 and not client.repair_events
    assert saved[0].ad_safety=='caution' and saved[0].ad_safety_verified is False
    assert saved[1].ad_safety_verified is True
    assert saved[0].post_priority==9


def test_safety_citation_must_belong_to_clip():
    ep=timeline()
    p=validate_candidate(ep,proposal())
    for reason in ['at E099_L015','at E001_L001','at E099_S001','notE099_L001fake']:
        assert not verify_safety(score(ad_safety='caution',ad_safety_reason=reason),ep,p).ad_safety_verified
    assert verify_safety(score(ad_safety='caution',ad_safety_reason='at [L001]'),ep,p).ad_safety_verified


def test_streamlit_demo_smoke_and_editor_decision(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm.LLMClient,'_request',lambda *a,**k:pytest.fail('Demo may not call a provider'))
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs]==['Run','Top promos','All candidates','Story bible','Timeline','Quality','How it works']
    assert any('promo-card' in element.proto.body for element in app.get('html'))
    button=next(button for button in app.button if button.key and button.key.startswith('approve_'))
    promo_id=button.key.removeprefix('approve_')
    button.click().run()
    assert not app.exception and app.session_state['decisions'][promo_id]['decision']=='approve'


def test_streamlit_upload_mode_and_clear_session(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm.LLMClient,'_request',lambda *a,**k:pytest.fail('No provider calls in UI smoke'))
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=30).run()
    app.radio[0].set_value('Run on my files').run()
    assert not app.exception
    assert any('Upload your episode' in message.value for message in app.info)
    next(field for field in app.text_input if field.label=='GROQ_API_KEY').set_value('synthetic-ui-key').run()
    workspace=new_workspace()
    root=Path(workspace.name)
    (root/'uploaded.txt').write_text('Synthetic private content')
    app.session_state['workspace']=workspace
    next(button for button in app.button if button.label=='Clear session').click().run()
    assert not app.exception and app.radio[0].value=='Demo'
    assert next(field for field in app.text_input if field.label=='GROQ_API_KEY').value==''
    assert not root.exists()


def test_cli_dry_run_preserves_files_and_options(tmp_path):
    import subprocess
    import sys
    import yaml
    source,config=setup_pipeline(tmp_path)
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config),encoding='utf-8')
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=subprocess.run([sys.executable,str(PROJECT_ROOT/'run.py'),'--input',str(source),
        '--tag','test','--config',str(path),'--episodes','all','--batch-size','5',
        '--max-api-calls','0','--candidate-provider','mock','--candidate-model','mock',
        '--judge-provider','mock','--judge-model','mock','--dry-run'],capture_output=True,text=True,encoding='utf-8',check=True)
    assert 'PIPELINE DRY RUN' in result.stdout and 'Shared call budget=0' in result.stdout
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_public_theme_and_confidential_ignores_exist():
    ignore=(PROJECT_ROOT/'.gitignore').read_text()
    assert all(name in ignore for name in ['data/','outputs/','parsed/','.cache/','.env','.streamlit/secrets.toml'])
    import tomllib
    theme=tomllib.loads((PROJECT_ROOT/'.streamlit/config.toml').read_text())['theme']
    assert theme['backgroundColor']=='#1C1B31' and theme['primaryColor']=='#F64668'


def test_demo_promos_have_distinct_hooks_cliffhangers_and_rationales():
    import json
    from pathlib import Path
    candidates = [item for path in Path('demo/precomputed/results/candidates_demo').glob('episode_*.json')
                  for item in json.loads(path.read_text(encoding='utf-8'))]
    scores = json.loads(Path('demo/precomputed/results/judged_demo/scores.json').read_text(encoding='utf-8'))
    for field in ('hook_text', 'cliffhanger_text'):
        assert len({item[field] for item in candidates}) == len(candidates)
    assert len({score['why_it_works'] for score in scores}) == len(scores)
