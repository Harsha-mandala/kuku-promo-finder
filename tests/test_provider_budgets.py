import json
from pathlib import Path

from src import candidates,judge
from src.candidates import score_candidates
from src.models import CandidateList
from src.judge import JudgeBatch
from src.parser import parse_file
from src.pipeline import run_pipeline
from test_judge_rank_pipeline import setup_pipeline,mock,score
from test_candidates import prepare_run,proposal
from test_context import synthetic_bible


def test_exhausted_candidate_provider_does_not_block_judge(tmp_path,monkeypatch):
    source,config=setup_pipeline(tmp_path)
    text=(source/'episode.txt').read_text(encoding='utf-8')
    (source/'other.txt').write_text(text.replace('Episode 99','Episode 100'),encoding='utf-8')
    episode=parse_file(source/'episode.txt')
    config=prepare_run(tmp_path,episode,synthetic_bible(),[99,100])
    config['models']={'candidates':{'provider':'gemini','model':'mock'},'bible':{'provider':'gemini','model':'mock'},'judge':{'provider':'groq','model':'mock'}}
    config['llm']={'provider_budgets':{'gemini':0,'groq':1}}
    folder=Path(config['output_dir'])/'candidates_test'
    folder.mkdir()
    candidate=proposal(start_shot_id='S001',end_shot_id='S005',hook_line_id='L002',cliffhanger_line_id='L010')
    saved=score_candidates(episode,CandidateList(episode=99,candidates=[candidate]))
    (folder/'episode_099.json').write_text(json.dumps([p.model_dump() for p in saved]),encoding='utf-8')
    generator=mock(tmp_path/'gen',[]);generator.provider_name='gemini'
    reviewer=mock(tmp_path/'review',[JudgeBatch([score()]).model_dump_json()]);reviewer.provider_name='groq'
    monkeypatch.setattr(candidates,'get_llm',lambda *a:generator)
    monkeypatch.setattr(judge,'get_llm',lambda *a:reviewer)
    result=run_pipeline(config,source,'test')
    assert not generator.calls and len(reviewer.calls)==1
    assert result['stopped']=='candidates' and not result['judge_stats']['pending']


def test_embers_body_and_button_contrast():
    def luminance(value):
        channels=[int(value[i:i+2],16)/255 for i in (1,3,5)]
        linear=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in channels]
        return sum(a*b for a,b in zip(linear,[.2126,.7152,.0722]))
    for a,b in [('#FFF4EE','#1C1B31'),('#FFF4EE','#2A2946'),('#1C1B31','#F64668')]:
        x,y=sorted([luminance(a),luminance(b)])
        assert (y+.05)/(x+.05)>=4.5
