"""Build the original fictional demo through the real pipeline and MockProvider."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.judge import JudgeBatch, JudgeScore, plan_judge
from src.llm import MockProvider
from src.models import Candidate, CandidateList, Character, StoryArc, StoryBible
from src.parser import parse_file
from src.pipeline import run_pipeline
from src.timeutils import format_timestamp
from src.validate import validate_candidate
from src.eval import run_eval
from src.models import Trailer, TrailerSegment, TrailerBatch
from src.trailers import TrailerJudgeScore, TrailerJudgeBatch
from src.editor_pass import EditorChoice, EditorOrder
from src.rank import rank_promos

TITLE='The Midnight Patent'
LOGLINES=[
    'Clockmaker Nera exposes the theft of her invention during a public unveiling, but the patent names someone else.',
    'A stolen tower key leads Nera to a recording, where her closest ally admits sending the warning.',
    'Nera interrupts the midnight auction with proof of the real inventor, until a buyer names an impossible price.',
    'At the bell tower, Nera plays the evidence and discovers a second voice behind the betrayal.',
]
BEATS=[
    [('NERA','That machine is mine. You stole every drawing.'),('IVO','Your signature was never on the patent.'),
     ('SABLE','The board bought your silence, not your loyalty.'),('NERA','Then whose name did you put on it?'),
     ('IVO','Your father signed the transfer himself.'),('NERA','He was missing before the machine was finished.'),
     ('SABLE','Open the sealed envelope if you want the truth.'),('NERA','Why is there a photograph of me inside?')],
    [('ORIN','That key opens the tower—or your grave.'),('NERA','Someone recorded the night my father vanished.'),
     ('IVO','A voice can be copied. A debt cannot.'),('NERA','Then why does the recording say your name?'),
     ('IVO','Give me the key, or the tower burns.'),('ORIN','I sent the warning. I could not let you go alone.'),
     ('NERA','You knew where he was all this time.'),('ORIN','What if he never wanted you to find him?')],
    [('NERA','Stop the auction. You are selling stolen work.'),('SABLE','The buyers already know who built it.'),
     ('ORIN','The original drawing has a second signature.'),('NERA','Who signed it after my father disappeared?'),
     ('SABLE','I will trade the patent for one name.'),('NERA','You cannot buy another person with my invention.'),
     ('ORIN','The buyer is standing behind that curtain.'),('NERA','Why is my father bidding against me?')],
    [('NERA','That voice on the recording is yours.'),('IVO','Play the rest before you decide who betrayed you.'),
     ('SABLE','The bell will erase the last minute of evidence.'),('NERA','Then which of you ordered it to ring?'),
     ('IVO','I hid the original beneath the bell.'),('NERA','There are two voices on this recording.'),
     ('SABLE','You were never supposed to hear the second one.'),('NERA','Who just locked the tower door?')],
]


def screenplay(number):
    rows=[f'EPISODE NUMBER: Episode {number}',f'EPISODE LOGLINE: {LOGLINES[number-1]}',
          'CONTINUITY NOTES: An original fictional demo about a stolen clockwork invention.',
          '', '--- SCRIPT START ---','','SCENE 01','Scene heading: INT. CLOCK TOWER – NIGHT',
          'Scene start: 00:00.00','Scene end: 00:40.00','','Actions and dialogue:','']
    widths=[[7,8,8,7,6,8,9,9,10,8,10,10],[12,12,12,12,7,7,7,7,8,8,9,9],[5,5,6,6,8,8,9,9,10,10,10,10],[10]*12][number-1]
    duration=sum(widths)
    rows[9]='Scene end: '+format_timestamp(duration)
    extra=[
        [('NERA','You sold my identity along with the machine.'),('ORIN','The workshop seal was broken from inside.'),('SABLE','Your portrait appears on every forged patent.'),('NERA','Who has been signing my name?')],
        [('NERA','You imprisoned him inside his own invention.'),('ORIN','The mechanism responds only to his voice.'),('SABLE','That knock came from beneath the floor.'),('NERA','Father, can you hear me through the machine?')],
        [('NERA','The highest bidder ordered my father killed.'),('ORIN','The ledger lists tonight as the final payment.'),('SABLE','The buyer has just raised a second paddle.'),('NERA','Whose life are you auctioning next?')],
        [('NERA','You erased my mother from the recording.'),('ORIN','There is another voice beneath the tower bell.'),('SABLE','The door can only be locked from outside.'),('NERA','Mother, why are you holding the key?')],
    ][number-1]
    for i,(speaker,text) in enumerate(BEATS[number-1]+extra):
        start=sum(widths[:i])
        camera=f'Close view of {speaker.title()} beside the clockwork display; the brass pendulum swings behind them.'
        if i==11:
            camera='Hold on Nera as she turns toward the locked door, still waiting for an answer.'
        rows += ['**CAMERA:**',f'Start: ({format_timestamp(start)}) | End: ({format_timestamp(start+widths[i])})',camera,'']
        if number==1 and i==0:
            rows += ['**ACTION:**','Start: (00:00.10) | End: (00:02.00)',
                     'Nera slaps Ivo across the cheek, stopping the public unveiling.','']
        rows += [f'**{speaker} (Urgent):**',f'Start: ({format_timestamp(start+.3)}) | End: ({format_timestamp(start+3.8)})',f'"{text}"','']
    rows += ['**SOUND FX:**',f'Start: ({format_timestamp(duration-1.2)}) | End: ({format_timestamp(duration)})','The tower bell begins a single unfinished note.','','--- SCRIPT END ---','']
    return '\n'.join(rows)


def build_demo(root=Path('demo')):
    root=Path(root)
    inputs=root/'episodes'
    inputs.mkdir(parents=True,exist_ok=True)
    for number in range(1,5):
        (inputs/f'episode_{number:03d}.txt').write_text(screenplay(number),encoding='utf-8')
    episodes=[parse_file(p) for p in sorted(inputs.glob('*.txt'))]
    characters=[]
    for name in ['Nera','Ivo','Sable','Orin']:
        first=next(line for ep in episodes for line in ep.lines if line.speaker==name.upper())
        characters.append(Character(name=name,aliases=[name.upper()],role='A participant in the stolen-patent conflict.',
            relationships='Connected through the invention and the tower recording.',confidence='high',evidence=first.id))
    bible=StoryBible(title_guess=TITLE,premise='A clockmaker fights to reclaim her invention before the midnight bell destroys the evidence.',
                     characters=characters,main_conflicts=['Ownership of the invention','The missing inventor and a hidden recording'],
                     arcs=[StoryArc(episodes='1-2',summary='The patent theft leads to a forbidden tower recording.'),
                           StoryArc(episodes='3-4',summary='The auction and the bell reveal a larger betrayal.')])
    lists=[]
    validated=[]
    for ep in episodes:
        candidates=[]
        for first,last in ([(0,3),(8,11)] if ep.number==4 else [(0,3),(4,7),(8,11)]):
            hook=min(ep.shots[first].lines,key=lambda line:line.start)
            cliff=next(line for line in ep.shots[last].lines if line.type=='dialogue')
            start_index=first-1 if ep.number==4 and first==8 else first
            candidates.append(Candidate(start_shot_id=ep.shots[start_index].id,end_shot_id=ep.shots[last].id,
                hook_line_id=hook.id,cliffhanger_line_id=cliff.id,hook_type='slap_or_physical' if hook.type=='action' else 'shocking_reveal',
                cliffhanger_type='unanswered_question',context_needed='',why_it_works='Generator draft: conflict opens and a question remains.',self_score=5 if first==0 else 4))
        lists.append(CandidateList(episode=ep.number,candidates=candidates))
        for index,c in enumerate(candidates,1):
            validated.append(validate_candidate(ep,c.model_dump(),index))
    config={'parsed_dir':str(root/'precomputed/episodes'),'output_dir':str(root/'precomputed/results'),
            'models':{step:{'provider':'mock','model':'mock'} for step in ('candidates','bible','judge','trailers','editor')},
            'llm':{'provider':'mock','model':'mock','max_retries':1,'max_api_calls_per_run':14,'max_request_tokens':30000},
            'promo':{'candidates_per_episode':5},'trailers':{'enabled':True,'count':1,'max_episode':15},'editor':{'enabled':True,'seed':20261106}}
    plans=plan_judge(validated,{e.number:e for e in episodes},config,'mock',3)
    answers=[]
    all_scores=[]
    variants=[(4,4,3),(5,4,5),(4,5,4),(3,4,4),(4,3,5),(5,5,4),
              (4,4,5),(3,5,4),(5,3,5),(5,4,4),(4,5,3)]
    for plan in plans:
        scores=[]
        for p in plan['promos']:
            hook_score,standalone_score,emotion_score=variants[len(all_scores)+len(scores)]
            caution=p.id in ('E001_C01','E002_C02')
            reason=f'Scripted slap at {p.hook_line_id}.' if p.id=='E001_C01' else 'A spoken threat of fire.' if caution else ''
            scores.append(JudgeScore(promo_id=p.id,hook_score=hook_score,
                cliffhanger_score=5 if p.id.endswith('C02') else 4,standalone_score=standalone_score,emotion_score=emotion_score,
                post_priority=9 if p.id=='E001_C01' else 8 if p.id.endswith('C02') else 7,
                ad_safety='caution' if caution else 'safe',ad_safety_reason=reason,
                weakest_point='The invention’s history stays deliberately mysterious.',
                one_line_pitch='A stolen invention turns a public accusation into a midnight mystery.',
                why_it_works=f'The opening “{p.hook_text}” puts the conflict on screen; “{p.cliffhanger_text}” leaves its consequence unanswered.'))
        all_scores.extend(scores)
        answers.append(JudgeBatch(scores).model_dump_json())
    first,last=episodes[0],episodes[2]
    hook=min(first.shots[0].lines,key=lambda line:line.start)
    cliff=next(line for line in last.shots[3].lines if line.type=='dialogue')
    trailer=Trailer(segments=[TrailerSegment(start_shot_id=first.shots[0].id,end_shot_id=first.shots[2].id),
                              TrailerSegment(start_shot_id=last.shots[2].id,end_shot_id=last.shots[3].id)],
                    hook_line_id=hook.id,cliffhanger_line_id=cliff.id,story='A stolen invention leads Nera to a second signature tied to her missing father.',
                    why_it_works='An explosive public theft accusation leads to an unanswered question about the true inventor.',spoiler_risk='low',self_score=5)
    trailer_score=TrailerJudgeScore(promo_id='T001',hook_score=5,cliffhanger_score=5,standalone_score=4,emotion_score=4,
        post_priority=9,coherence=4,ad_safety='caution',ad_safety_reason=f'Scripted slap at {hook.id}.',weakest_point='The auction jump needs a clear visual cut.',
        one_line_pitch='A stolen invention has a second author.',why_it_works='The slap stops the unveiling; the second signature opens a new mystery.')
    singles=rank_promos(validated,all_scores,config)['ranked']
    ids=[row['promo_id'] for row in singles]
    by_id={row['promo_id']:row for row in singles}
    editor_answer=EditorOrder([EditorChoice(promo_id=key,reason=f'The opening “{by_id[key]["hook_text"]}” states the conflict; “{by_id[key]["cliffhanger_text"]}” gives a new viewer a concrete question.') for key in ids]+[
        EditorChoice(promo_id='T001',reason='The mystery is strong, but the auction jump needs more context than these singles.')])
    extra_answers=[TrailerBatch([trailer]).model_dump_json(),TrailerJudgeBatch([trailer_score]).model_dump_json(),editor_answer.model_dump_json()]
    with tempfile.TemporaryDirectory(prefix='kuku_demo_build_') as logs:
        config['llm'].update(cache_dir=str(Path(logs)/'cache'),raw_log_dir=str(Path(logs)/'raw'))
        client=MockProvider(responses=[bible.model_dump_json(),json.dumps([c.model_dump() for c in lists])]+answers+extra_answers,
                            cache_dir=Path(logs)/'cache',raw_log_dir=Path(logs)/'raw')
        client.limiter.wait=lambda *args:None
        with patch('src.candidates.get_llm',return_value=client),patch('src.judge.get_llm',return_value=client),patch('src.editorial_io.get_llm',return_value=client):
            result=run_pipeline(config,inputs,'demo',batch_size=4,max_api_calls=14,force=True)
        if result.get('stopped'):
            raise RuntimeError('Demo generation did not finish.')
    (root/'provenance.json').write_text(json.dumps({'title':TITLE,'fictional':True,'provider':'MockProvider','live_api_calls':0,
        'episodes':4,'promos':11,'trailers':1,'editor_pass':True,'recipe':'python -m demo.build_demo'},indent=2),encoding='utf-8')
    config['evaluation']={'gold_csv':str(root/'gold.csv'),'gold_episodes':[1,2,3,4],
                          'seed':20261106,'bootstrap_samples':2000}
    run_eval(config,'demo',eval_dir=root/'precomputed/evaluation')
    return result


if __name__=='__main__':
    build_demo()
