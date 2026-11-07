import json
from pathlib import Path

import pytest

from src.baselines import windows
from src.editor_pass import EditorChoice, EditorOrder, blend_ranks, run_editor
from src.eval import load_saved
from src.eval_metrics import intersection_over_union, matches
from src.eval_strict import chance_levels, strict_evaluation
from src.llm import CallBudget, LLMBudgetError, MockProvider
from src.models import Trailer, TrailerBatch, TrailerSegment
from src.trailers import TrailerJudgeBatch, TrailerJudgeScore, hard_errors, render_trailer, run_trailers, validate_trailer, rank_trailers
from test_eval import make_saved, pick


def setup(tmp_path):
    config=make_saved(tmp_path)
    config['models'].update(trailers={'provider':'mock','model':'mock'},editor={'provider':'mock','model':'mock'},judge={'provider':'mock','model':'mock'})
    config['llm']={'max_retries':1,'max_api_calls_per_run':14}
    config['trailers']={'enabled':True,'count':5,'max_episode':15}
    config['editor']={'enabled':True,'seed':7}
    final=Path(config['output_dir'])/'final_test'
    rows=json.loads((final/'promos_ranked.json').read_text(encoding='utf-8'))
    for row in rows:
        row.update(status='ok',ad_safety='safe',duration_s=20,cliffhanger_score=4,score_adj=row['final_score'])
    (final/'promos_ranked.json').write_text(json.dumps(rows),encoding='utf-8')
    (final/'report.md').write_text('# Synthetic report\n',encoding='utf-8')
    return config,load_saved(config,'test')['episodes']


def proposal(second=2):
    return Trailer(segments=[TrailerSegment(start_shot_id='E001_S001',end_shot_id='E001_S003'),
                             TrailerSegment(start_shot_id=f'E{second:03d}_S006',end_shot_id=f'E{second:03d}_S008')],
                   hook_line_id='E001_L001',cliffhanger_line_id=f'E{second:03d}_L008',story='Synthetic setup to unanswered question.',
                   why_it_works='Synthetic escalating conflict.',spoiler_risk='low',self_score=5)


def score(key='T001'):
    return TrailerJudgeScore(promo_id=key,hook_score=4,cliffhanger_score=4,standalone_score=4,emotion_score=4,
                             post_priority=8,coherence=5,ad_safety='safe',ad_safety_reason='',weakest_point='Synthetic gap.',
                             one_line_pitch='Synthetic trailer.',why_it_works='Synthetic hook and final unanswered question.')


def mock(tmp_path,replies):
    client=MockProvider(responses=replies,max_retries=1,cache_dir=tmp_path/'cache',raw_log_dir=tmp_path/'raw')
    client.limiter.wait=lambda *args:None
    return client


def test_iou_math_and_hook_edges():
    assert intersection_over_union((0,20),(10,30))==pytest.approx(1/3)
    assert intersection_over_union((0,20),(0,20))==1
    assert intersection_over_union((0,20),(20,40))==0
    assert matches(pick(1,2.9,22.9),pick(1,0,20),rule='hook')
    assert not matches(pick(1,3.1,23.1),pick(1,0,20),rule='hook')
    assert not matches(pick(1,2,4),pick(1,0,20),rule='hook')
    assert not matches(pick(2,0,20),pick(1,0,20),rule='hook')


def test_chance_is_exact_enumeration_and_random_100_seeds_repeat(tmp_path):
    _,episodes=setup(tmp_path)
    gold=[pick(1,0,20)]
    chance=chance_levels(gold,episodes)
    valid=list(windows(episodes[1],20,60))
    expected=sum(matches(pick(1,a,b),gold[0],rule='iou') for _,_,a,b in valid)/len(valid)
    assert chance['iou']['hit_chance']==expected
    inputs={key:[] for key in ('random','rules_only','generator_self_score','full_pipeline')}
    first=strict_evaluation(inputs,gold,episodes,seed=7,samples=20)
    second=strict_evaluation(inputs,gold,episodes,seed=7,samples=20)
    assert first==second and len(first['random_seeds'])==100
    assert first['primary']['random']['iou']['n_seeds']==100
    assert 0<=first['primary']['random']['coverage']<=1


@pytest.mark.parametrize('second',[1,2])
def test_valid_trailer_same_or_cross_episode(tmp_path,second):
    _,episodes=setup(tmp_path)
    final=validate_trailer(proposal(second),episodes)
    assert final.status=='ok' and final.duration_s==30
    assert '--- CUT ---' in render_trailer(final,episodes)
    assert final.segments[1].episode==second


def test_overlap_rejected_and_single_segment_cannot_cross_episode(tmp_path):
    _,episodes=setup(tmp_path)
    bad=proposal(1)
    bad.segments[1]=TrailerSegment(start_shot_id='E001_S003',end_shot_id='E001_S005')
    bad.cliffhanger_line_id='E001_L005'
    assert validate_trailer(bad,episodes).status=='rejected'
    bad.segments[1].end_shot_id='E002_S005'
    assert hard_errors(bad,episodes)


def test_adjacent_trailer_segments_are_rejected(tmp_path):
    _, episodes = setup(tmp_path)
    item = proposal(1)
    item.segments[1].start_shot_id = 'E001_S004'
    final = validate_trailer(item, episodes)
    assert final.status == 'rejected'
    assert final.fix_log[-1] == 'rejected: same-episode segment gap 0 s below 10 s'


def test_total_over_sixty_trims_only_last_segment(tmp_path):
    _,episodes=setup(tmp_path)
    item=proposal()
    item.segments[0].end_shot_id='E001_S005'
    item.segments[1]=TrailerSegment(start_shot_id='E002_S001',end_shot_id='E002_S010')
    item.cliffhanger_line_id='E002_L006'
    final=validate_trailer(item,episodes)
    assert final.duration_s==60 and final.status=='fixed'
    assert final.segments[0].end_shot_id=='E001_S005'
    assert final.segments[1].end_shot_id=='E002_S007'


def test_trailer_wrong_hook_segment_repairs_once_and_resume_no_calls(tmp_path):
    config,_=setup(tmp_path)
    item=proposal();bad=item.model_copy(deep=True);bad.hook_line_id='E002_L006'
    generator=mock(tmp_path/'gen',[TrailerBatch([bad]).model_dump_json(),TrailerBatch([item]).model_dump_json()])
    judge=mock(tmp_path/'judge',[TrailerJudgeBatch([score()]).model_dump_json()])
    result=run_trailers(config,'test',client=generator,judge_client=judge)
    assert len(generator.calls)==2 and len(judge.calls)==1 and len(result['trailers'])==1
    assert '--- CUT ---' in judge.calls[0].user
    assert judge.calls[0].temperature==0
    assert 'self_score' not in generator.calls[0].user.split('Return ONLY valid JSON')[0]
    empty=mock(tmp_path/'empty',[])
    again=run_trailers(config,'test',client=empty,judge_client=empty)
    assert not empty.calls and again['trailers']==result['trailers']


def test_trailer_budget_is_shared_and_does_not_send(tmp_path):
    config,_=setup(tmp_path)
    client=mock(tmp_path/'mock',[])
    with pytest.raises(LLMBudgetError):
        run_trailers(config,'test',client=client,budget=CallBudget(0))
    assert not client.calls


@pytest.mark.parametrize('bad_kind',['missing','duplicate','unknown'])
def test_editor_fallback_repair_and_blend(tmp_path,bad_kind):
    config,_=setup(tmp_path)
    ids=[r['promo_id'] for r in json.loads((Path(config['output_dir'])/'final_test/promos_ranked.json').read_text())]
    good=[EditorChoice(promo_id=key,reason='Synthetic posting reason.') for key in reversed(ids)]
    bad=list(good[:-1])
    if bad_kind=='duplicate':bad.append(good[0])
    if bad_kind=='unknown':bad.append(EditorChoice(promo_id='unknown',reason='Unexpected ID.'))
    client=mock(tmp_path/'edit',[EditorOrder(bad).model_dump_json(),EditorOrder(good).model_dump_json()])
    result=run_editor(config,'test',client=client)
    assert len(client.calls)==(2 if bad_kind=='unknown' else 1)
    assert 'self_score' not in client.calls[0].user and 'final_score' not in client.calls[0].user
    assert sorted(row['final_rank'] for row in result['order'])==list(range(1,len(ids)+1))
    expected = [] if bad_kind=='unknown' else [good[-1].promo_id]
    assert result['fallback']['missing_appended'] == expected
    duplicates = [good[0].promo_id] if bad_kind=='duplicate' else []
    assert result['fallback']['duplicates_dropped'] == duplicates
    assert {row['promo_id'] for row in result['order'] if row['editor_fallback']} == set(expected + duplicates)
    source=Path(config['output_dir'])/'final_test/promos_ranked.json'
    source.write_text(json.dumps(list(reversed(json.loads(source.read_text())))),encoding='utf-8')
    empty=mock(tmp_path/'empty',[])
    run_editor(config,'test',client=empty,budget=CallBudget(0))
    assert not empty.calls


def test_rank_blending_known_numbers():
    rows=[{'promo_id':'A','final_score':.9,'cliffhanger_score':4,'duration_s':20},
          {'promo_id':'B','final_score':.8,'cliffhanger_score':4,'duration_s':20}]
    result=blend_ranks(rows,[EditorChoice(promo_id='B',reason='First.'),EditorChoice(promo_id='A',reason='Second.')])
    assert result[0]['promo_id']=='A' and result[0]['final_rank_score']==pytest.approx(.6)
    assert result[1]['final_rank_score']==pytest.approx(.4)


def test_clean_boundary_fixes_per_segment_keep_source_ids(tmp_path):
    _,episodes=setup(tmp_path)
    episodes[2].shots[5].clean_start=False
    final=validate_trailer(proposal(),episodes)
    assert final.status=='fixed'
    assert final.segments[1].start_shot_id=='E002_S005'
    assert any('unsafe start moved back' in entry for entry in final.fix_log)


def test_hidden_sound_line_id_is_not_allowed_from_pool(tmp_path):
    _,episodes=setup(tmp_path)
    episodes[2].shots[7].lines[0].type='sound'
    errors=hard_errors(proposal(),episodes)
    assert errors and 'not supplied' in errors[0]


def test_three_segment_trailer_and_soft_diversity_ranking(tmp_path):
    config,episodes=setup(tmp_path)
    item=proposal()
    item.segments.append(TrailerSegment(start_shot_id='E007_S001',end_shot_id='E007_S003'))
    item.cliffhanger_line_id='E007_L003'
    final=validate_trailer(item,episodes)
    assert final.duration_s==45 and final.status=='ok'
    assert render_trailer(final,episodes).count('--- CUT ---')==2
    other=validate_trailer(proposal(1),episodes,index=2)
    ranked=rank_trailers([final,other],[score(),score('T002')],config)
    assert [r['rank'] for r in ranked]==[1,2]
    assert ranked[1]['score_adj']==pytest.approx(ranked[1]['final_score']-.05)


def test_editor_and_trailer_judge_use_same_budget_and_save_progress(tmp_path):
    config,_=setup(tmp_path)
    budget=CallBudget(2)
    generator=mock(tmp_path/'g',[TrailerBatch([proposal()]).model_dump_json()])
    reviewer=mock(tmp_path/'r',[TrailerJudgeBatch([score()]).model_dump_json()])
    run_trailers(config,'test',client=generator,judge_client=reviewer,budget=budget,judge_budget=budget)
    assert budget.used==2
    editor=mock(tmp_path/'e',[])
    with pytest.raises(LLMBudgetError):
        run_editor(config,'test',client=editor,budget=budget)
    assert not editor.calls
    assert (Path(config['output_dir'])/'final_test/trailers_ranked.json').exists()


def test_editorial_dry_run_flags_and_force_episode_are_read_only(monkeypatch):
    from src.pipeline import plan_pipeline
    from src.ui.data import DEMO_ROOT,load_demo
    from src import editorial_io
    monkeypatch.setattr(editorial_io,'get_llm',lambda *a,**k:pytest.fail('Dry run must never create a client'))
    bundle=load_demo()
    config={'parsed_dir':str(DEMO_ROOT/'precomputed/episodes'),'output_dir':str(bundle['output']),
            'models':{step:{'provider':'mock','model':'mock'} for step in ('candidates','bible','judge','trailers','editor')},
            'promo':{'candidates_per_episode':5},'trailers':{'enabled':True,'count':1},'editor':{'enabled':True}}
    result=plan_pipeline(config,'demo',bundle['episodes'],bundle['episodes'],5,force_episodes={3})
    assert result['candidate_requests']==1
    assert {r['stage'] for r in result['editorial_stages']}=={'trailers','trailer_judge','editor'}
    disabled=plan_pipeline(config,'demo',bundle['episodes'],bundle['episodes'],5,no_trailers=True,no_editor=True)
    assert not disabled['editorial_stages']


def test_human_order_correlations_use_higher_is_better_negative_rank(tmp_path):
    import csv
    from src.eval import human_agreement
    config,_=setup(tmp_path)
    output=Path(config['output_dir'])
    ids=['E001_C01','E001_C02','E001_C03']
    rows=[{'promo_id':key,'judge_rank':i,'editor_rank':4-i,'final_rank':i} for i,key in enumerate(ids,1)]
    (output/'final_test/editor_order.json').write_text(json.dumps(rows),encoding='utf-8')
    path=tmp_path/'to_rate_test.csv'
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['promo_id','my_rating']);writer.writeheader()
        writer.writerows({'promo_id':key,'my_rating':4-i} for i,key in enumerate(ids,1))
    result=human_agreement(path,load_saved(config,'test'))
    assert result['judge_order']['spearman']==pytest.approx(1)
    assert result['editor_order']['spearman']==pytest.approx(-1)
    assert result['final_order']['spearman']==pytest.approx(1)


def test_ui_final_ranks_remain_unique_for_unedited_singles(tmp_path):
    from src.ui.data import load_bundle
    config,_=setup(tmp_path)
    final=Path(config['output_dir'])/'final_test'
    singles=json.loads((final/'promos_ranked.json').read_text())
    trailer=dict(singles[0],promo_id='T001',type='trailer',rank=1)
    edited=[dict(trailer,judge_rank=1,editor_rank=1,final_rank=1),
            dict(singles[0],type='single',judge_rank=2,editor_rank=2,final_rank=2)]
    (final/'trailers_ranked.json').write_text(json.dumps([trailer]),encoding='utf-8')
    (final/'editor_order.json').write_text(json.dumps(edited),encoding='utf-8')
    bundle=load_bundle(config['parsed_dir'],config['output_dir'],'test')
    ranks=[row['rank'] for row in bundle['ranked']+bundle['trailers']]
    assert sorted(ranks)==list(range(1,len(ranks)+1))


def test_stale_editor_order_is_ignored_after_source_changes(tmp_path):
    from src.editorial_io import editor_source_key,fresh_editor_rows
    config,_=setup(tmp_path)
    output=Path(config['output_dir']);final=output/'final_test'
    (final/'editor_order.json').write_text(json.dumps([{'promo_id':'E001_C01','final_rank':1,'type':'single'}]),encoding='utf-8')
    key=editor_source_key(output,'test',config['parsed_dir'])
    (final/'editor_inputs.json').write_text(json.dumps({'key':'synthetic','source_key':key}),encoding='utf-8')
    assert fresh_editor_rows(output,'test',config['parsed_dir'])
    rows=json.loads((final/'promos_ranked.json').read_text())
    rows[0]['segments'][0]['end']=25
    (final/'promos_ranked.json').write_text(json.dumps(rows),encoding='utf-8')
    assert fresh_editor_rows(output,'test',config['parsed_dir'])==[]
    assert load_saved(config,'test')['editor_rows']==[]
