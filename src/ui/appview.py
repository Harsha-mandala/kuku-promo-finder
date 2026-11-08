"""Streamlit presentation; all execution delegates to the shared pipeline."""

import hashlib
import json
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from ..cliprender import render_clip
from ..pipeline import parse_inputs, run_pipeline
from ..safety_review import shortlist
from ..timeutils import format_timestamp
from .data import DEMO_ROOT, PROJECT_ROOT, existing_run_tags, load_bundle, load_demo, local_runs_allowed, upload_config
from .helpers import filter_promos, redact, render_html, segment_text
from .session import clear_workspace, new_workspace, resolve_credentials, unpack_uploads, upload_fingerprint, write_uploads
from .theme import CSS, card_html
from .quality import show_quality
from .trailers_view import show_trailers
from .downloads import show_downloads
from .guide import show_guide

TABS=['Run','Top promos','All candidates','Story bible','Timeline','Quality','How it works']
STAGES=['parse','bible','candidates','validate','judge','rank','trailers','editor']


def reset_session():
    epoch=st.session_state.get('ui_epoch',0)+1
    if st.session_state.get('workspace'):
        clear_workspace(st.session_state['workspace'])
    st.session_state.clear()
    st.session_state['ui_epoch']=epoch


def model_controls(config,step,label):
    epoch=st.session_state.get('ui_epoch',0)
    current=config['models'][step]
    providers=list(dict.fromkeys([current['provider'],'gemini','groq','openai','ollama','mock']))
    provider=st.selectbox(f'{label} provider',providers,key=f'{step}_provider_{epoch}',
                          format_func=lambda name: 'mock — test run, no API key, fake scores' if name=='mock' else name)
    models=list(dict.fromkeys([v['model'] for v in config['models'].values() if v['provider']==provider]))
    if not models:
        models=['mock' if provider=='mock' else 'custom-model']
    selected=st.selectbox(f'{label} model',models,key=f'{step}_model_choice_{provider}_{epoch}')
    model=st.text_input(f'Edit {label.lower()} model',selected,key=f'{step}_model_{provider}_{selected}_{epoch}')
    return provider,model


def show_estimate(plan):
    rows=[{'Stage':'Bible','Requests':plan.get('bible',{}).get('requests',0),'Estimated input tokens':plan.get('bible',{}).get('tokens',0)},
          {'Stage':'Candidates','Requests':plan.get('candidate_requests',0),'Estimated input tokens':plan.get('candidate_tokens',0)},
          {'Stage':'Judge','Requests':plan.get('judge_requests',0),'Estimated input tokens':plan.get('judge_tokens',0)}]
    rows.extend({'Stage':row['stage'],'Requests':row['requests'],'Estimated input tokens':row['tokens']} for row in plan.get('editorial_stages',[]))
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('Input-token estimate only. Future judge selections are unknown; response tokens, retries and repairs are excluded.')


def show_script(episode,promo):
    rows=[]
    for line in render_clip(episode,promo).splitlines():
        hook=promo.hook_line_id and f'[{promo.hook_line_id}]' in line
        cliff=promo.cliffhanger_line_id and f'[{promo.cliffhanger_line_id}]' in line
        kind='both' if hook and cliff else 'hook' if hook else 'cliff' if cliff else ''
        label='HOOK + CLIFFHANGER' if hook and cliff else 'HOOK' if hook else 'CLIFFHANGER' if cliff else ''
        rows.append(f'<div class="script-line {kind}">'+(f'<div class="script-label">{label}</div>' if label else '')+escape(line)+'</div>')
    st.html(render_html(''.join(rows)))


def show_promos(bundle):
    rows=bundle['ranked']
    if not rows:
        st.info('No ranked promos yet. Estimate and run your input pack in the Run tab.')
        return
    lo,hi=min(r['episode'] for r in rows),max(r['episode'] for r in rows)
    columns=st.columns([2,2,1,2])
    with columns[0]:
        rng=st.slider('Episode range',lo,hi,(lo,hi)) if lo<hi else (lo,hi)
    with columns[1]:
        minimum=st.slider('Minimum score',-0.25,1.0,0.0,0.05)
    with columns[2]:
        hide=st.checkbox('Hide caution')
    with columns[3]:
        statuses=st.multiselect('Status',['ok','fixed','flagged'],default=['ok','fixed','flagged'])
    queue = bundle.get('safety_queue', [])
    if queue:
        with st.expander(f'Needs safety review ({len(queue)})'):
            for item in queue:
                st.write(f"{item['promo_id']} · {item['ad_safety']} · {item['reason']}")
    review_only = st.checkbox('Needs safety review only', disabled=not queue)
    safe_only = st.toggle('Safe-only top 10', disabled=review_only)
    filtered = filter_promos(rows, rng, minimum, hide and not review_only, statuses)
    visible_trailers = filter_promos([row for row in bundle.get('trailers', []) if row['ad_safety'] != 'unsafe'],
                                    rng, minimum, hide and not review_only, statuses)
    automatic = st.toggle('Automatic top 10', disabled=review_only or safe_only)
    if review_only:
        filtered = [row for row in filtered if row.get('safety_review')]
        visible_trailers = [row for row in visible_trailers if row.get('safety_review')]
    else:
        filtered = [row for row in filtered if not row.get('safety_review')]
        visible_trailers = [row for row in visible_trailers if not row.get('safety_review')]
        if safe_only or automatic:
            eligible = [row for row in filtered + visible_trailers if not safe_only or row['ad_safety'] == 'safe']
            eligible.sort(key=lambda row: row.get('final_rank', float('inf')))
            top = shortlist(eligible)
            filtered = [row for row in top if row.get('type') != 'trailer']
            visible_trailers = [row for row in top if row.get('type') == 'trailer']
    st.caption(f'{len(filtered)} singles · {len(visible_trailers)} trailers · Magenta marks the hook; coral marks the cliffhanger. Scores support an editor’s judgment.')
    episodes={e.number:e for e in bundle['episodes']}
    promos={p.id:p for p in bundle['validated']}
    decisions=st.session_state.setdefault('decisions',{})
    st.html(render_html("""<script>
    if(!window.kukuCopyHandler){document.addEventListener('click',async e=>{const b=e.target.closest('button.copy-segment');if(b){try{await navigator.clipboard.writeText(b.dataset.segment);b.textContent='Copied';}catch(error){const t=document.createElement('textarea');t.value=b.dataset.segment;document.body.appendChild(t);t.select();const ok=document.execCommand('copy');t.remove();b.textContent=ok?'Copied':'Copy unavailable';}}});window.kukuCopyHandler=true;}
    </script>"""),unsafe_allow_javascript=True)
    show_trailers(dict(bundle,trailers=visible_trailers),episodes)
    for i,row in enumerate(filtered):
        st.html(card_html(row))
        for alternate in row.get('alternate_cuts', []):
            st.caption(f"{alternate['promo_id']} · alternate cut of {row['promo_id']}")
        if row.get('ad_safety_reason'):
            st.caption('Safety: '+row['ad_safety_reason']+(' · unverified source citation' if not row.get('ad_safety_verified',True) else ''))
        if row.get('formula_note','').startswith('old'):
            st.caption(row['formula_note'])
        promo=promos.get(row['promo_id'])
        with st.expander('Script of this clip',expanded=i==0 and st.query_params.get('expand')=='1'):
            if promo:
                show_script(episodes[promo.episode],promo)
        with st.expander('What the validator changed'):
            if promo:
                original='–'.join(format_timestamp(t) for t in promo.orig_span) if promo.orig_span else 'Unavailable'
                final='–'.join(format_timestamp(t) for t in promo.final_span) if promo.final_span else 'Unavailable'
                st.write(f'Original: {original} → Final: {final}')
                st.write('\n'.join('• '+item for item in promo.fix_log) or 'No boundary changes.')
        note=st.text_input('Editor note',key=f"note_{st.session_state.get('ui_epoch',0)}_{bundle['decision_scope']}_{row['promo_id']}")
        approve,reject,decision=st.columns([1,1,3])
        for column,label,value in [(approve,'✅ Approve','approve'),(reject,'❌ Reject','reject')]:
            if column.button(label,key=value+'_'+row['promo_id'],width='stretch'):
                decisions[row['promo_id']]={'decision':value,'note':note}
                st.rerun()
        if row['promo_id'] in decisions:
            decision.caption('Editor decision: '+decisions[row['promo_id']]['decision'])
    if not filtered:
        st.info('No promos match these filters.')


def all_candidates(bundle):
    validated={p.id:p for p in bundle['validated']}
    scores={r['promo_id']:r for r in bundle['scores']}
    rows=[]
    for path in sorted((bundle['output']/f"candidates_{bundle['tag']}").glob('episode_*.json')):
        for i,raw in enumerate(json.loads(path.read_text(encoding='utf-8')),1):
            key=raw.get('candidate_id',f"E{int(path.stem.split('_')[-1]):03d}_C{i:02d}")
            promo=validated.get(key)
            score=scores.get(key,{})
            rows.append({'Promo':key,'Episode':promo.episode if promo else int(path.stem.split('_')[-1]),
                'Status':promo.status if promo else 'pending validation','Safety':score.get('ad_safety','not judged'),
                'Verified':score.get('ad_safety_verified'), 'Duration (s)':promo.duration_s if promo else raw.get('duration_s'),
                'Self score':raw.get('self_score'),'Post priority':score.get('post_priority'),
                'Hook':promo.hook_text if promo else raw.get('hook_text',''),
                'Fixes / rejection reason':' | '.join(promo.fix_log) if promo else 'Not validated yet'})
    if not rows:
        st.info('Candidates appear here after generation.')
        return
    columns = st.multiselect('Table columns', list(rows[0]),
        default=['Promo', 'Status', 'Safety', 'Duration (s)', 'Hook', 'Fixes / rejection reason'])
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', column_order=columns,
        column_config={'Hook': st.column_config.TextColumn(width=180),
                       'Fixes / rejection reason': st.column_config.TextColumn(width=220)})
    st.caption('Click a column heading to sort. Unsafe and rejected candidates remain visible here.')


def show_bible(bundle):
    bible=bundle['bible']
    if not bible:
        st.info('The story bible appears after context generation.')
        return
    st.subheader(bible.get('title_guess','Story bible'))
    st.write(bible.get('premise',''))
    records=[{k:c.get(k,'') for k in ['name','aliases','confidence','evidence']} for c in bible.get('characters',[])]
    st.dataframe(pd.DataFrame(records),hide_index=True,width='stretch')
    if bible.get('minor_characters'):
        st.caption('Other characters: '+', '.join(bible['minor_characters']))
    st.subheader('Story arcs')
    for arc in bible.get('arcs',[]):
        st.markdown(f"**Episodes {arc['episodes']}** — {arc['summary']}")


def show_timeline(bundle):
    st.html(render_html('<div class="section-intro">Every selection spans complete shots. Hover over a block for timestamps.</div>'))
    st.html(render_html('<div class="legend"><span>● Coral: ranks 1–3</span><span>● Pink: other promos</span><span>Dashed: trailer segments</span></div>'))
    for ep in bundle['episodes']:
        if ep.duration <= 0:
            continue
        bars=[]
        for row in bundle['ranked']:
            for segment in row['segments']:
                if segment['episode']!=ep.number or ep.duration<=0:
                    continue
                left=100*segment['start']/ep.duration
                width=100*(segment['end']-segment['start'])/ep.duration
                kind='gold' if row['rank']<=3 else ''
                bars.append(f'<div class="timeline-span {kind}" style="left:{left:.2f}%;width:{width:.2f}%" title="{escape(segment_text(row),quote=True)}">#{row["rank"]}</div>')
        for row in bundle.get('trailers', []):
            for index, segment in enumerate(row['segments'], 1):
                if segment['episode'] != ep.number:
                    continue
                left = 100 * segment['start'] / ep.duration
                width = 100 * (segment['end'] - segment['start']) / ep.duration
                label = f"{row['promo_id']} · {index}"
                times = f"{format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}"
                tooltip = escape(f'{label} · {times}', quote=True)
                bars.append(f'<div class="timeline-span trailer-span" style="left:{left:.2f}%;width:{width:.2f}%" title="{tooltip}">{escape(label)}</div>')
        st.html(render_html(f'<div class="timeline-row"><strong>Episode {ep.number:02d}</strong> <span class="meta"> / {format_timestamp(ep.duration)}</span><div class="meta">{escape(ep.logline)}</div><div class="timeline-track">'+''.join(bars)+'</div></div>'))


def main():
    st.set_page_config(page_title='Kuku Promo Finder',page_icon='🎬',layout='wide')
    st.html(render_html(CSS))
    allow_existing=local_runs_allowed(os.environ, st.context.headers)
    config_override = os.environ.get('KUKU_CONFIG') if allow_existing else None
    config_path = PROJECT_ROOT / (config_override or 'config.yaml')
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    epoch=st.session_state.setdefault('ui_epoch',0)
    project_output=PROJECT_ROOT/config.get('output_dir','outputs')
    modes=['Demo','Run on my files']
    if allow_existing:
        modes.append('Open existing run')
    with st.sidebar:
        st.html(render_html('<div class="eyebrow">KUKU / PROMO STUDIO</div>'))
        mode=st.radio('Mode',modes,key=f'mode_{epoch}')
        existing=allow_existing and mode=='Open existing run'
        run_tag=None
        if existing:
            tags=existing_run_tags(project_output)
            if tags:
                run_tag=st.selectbox('Saved run',tags,index=tags.index('gemini_full') if 'gemini_full' in tags else 0)
            st.caption('Local results are opened read-only.')
        st.caption('Demo is fictional, precomputed and key-free.' if mode=='Demo' else 'Uploads and outputs stay in this session’s temporary folder.' if not existing else 'Browse saved results on this computer.')
        uploads=st.file_uploader('Input pack · TXT or ZIP',type=['txt','zip'],accept_multiple_files=True,disabled=mode!='Run on my files',key=f'uploads_{epoch}')
        with st.expander('Models',expanded=mode=='Run on my files'):
            cp,cm=model_controls(config,'candidates','Candidate')
            jp,jm=model_controls(config,'judge','Judge')
            if mode=='Run on my files':
                for stage in ('trailers','editor'):
                    if config.get(stage,{}).get('enabled',False):
                        setting=config['models'][stage]
                        st.caption(f"Configured {stage}: {setting['provider']} / {setting['model']}")
        with st.expander('API keys · optional'):
            gemini=st.text_input('GEMINI_API_KEY',type='password',key=f'gemini_key_{epoch}',disabled=mode!='Run on my files')
            groq=st.text_input('GROQ_API_KEY',type='password',key=f'groq_key_{epoch}',disabled=mode!='Run on my files')
            st.caption('Leave blank to use app secrets or environment keys. Keys are never saved to disk.')
        batch=st.number_input('Batch size',min_value=1,max_value=20,value=5,key=f'batch_{epoch}')
        calls=st.number_input('Max API calls',min_value=0,max_value=1000,value=14,key=f'calls_{epoch}')
        selector=st.text_input('Episodes','all',help='Examples: all, 1-10, 1,3,5',key=f'selector_{epoch}')
        estimate_clicked=st.button('Estimate cost',width='stretch',disabled=existing)
        st.button('Clear session',width='stretch',on_click=reset_session)
    st.html(render_html('<h1 class="royal-title">Kuku Promo Finder</h1><div class="royal-subtitle">Find the moment that makes them watch the next episode.</div>'))
    keys={}
    hosted=False
    bundle=load_demo()
    active_config=None
    input_dir=None
    ready=mode=='Demo'
    mock_upload=mode=='Run on my files' and 'mock' in (cp,jp)
    upload_tag='session_mock' if mock_upload else 'session'
    if mock_upload:
        cp=jp=cm=jm='mock'
    if mode=='Run on my files':
        bundle={'ranked':[],'episodes':[],'validated':[],'scores':[],'bible':{},'exclusions':{},'output':DEMO_ROOT/'unloaded','tag':'session'}
        try:
            secrets=st.secrets.to_dict()
        except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):
            secrets={}
        providers={cp,jp,config['models']['bible']['provider']}
        providers.update(config['models'][stage]['provider'] for stage in ('trailers','editor') if config.get(stage,{}).get('enabled',False))
        keys,hosted=resolve_credentials({'GEMINI_API_KEY':gemini,'GROQ_API_KEY':groq},secrets,os.environ,providers)
        if mock_upload:
            keys,hosted={},False
            st.info('Mock upload run. All stages use test responses with fake scores.')
        if hosted:
            calls=min(14,calls)
            st.sidebar.caption('Hosted-key limits: 20 episodes and 14 API calls per run.')
        if uploads:
            try:
                files=unpack_uploads([(file.name,file.getvalue()) for file in uploads])
                signature=upload_fingerprint(files)
                if st.session_state.get('upload_signature')!=signature:
                    if st.session_state.get('workspace'):
                        clear_workspace(st.session_state['workspace'])
                    st.session_state['workspace']=new_workspace()
                    st.session_state['upload_signature']=signature
                    st.session_state['decisions']={}
                    st.session_state.pop('estimate',None)
                root=Path(st.session_state['workspace'].name)/('mock' if mock_upload else 'live')
                input_dir=write_uploads(root,files)
                active_config=upload_config(config,root,keys,mock_upload)
                preview=parse_inputs(input_dir,root/'parsed',dry_run=True)
                if hosted and len(preview)>20:
                    raise ValueError('Hosted runs are capped at 20 episodes / 14 calls. Paste your own API keys or use the CLI for a full show.')
                bundle=load_bundle(root/'parsed',root/'outputs',upload_tag)
                bundle['episodes']=preview
                ready=True
            except (OSError,ValueError) as error:
                st.error(redact(error,keys))
        else:
            st.info('Upload your episode TXT files or a ZIP input pack to begin.')
    if existing:
        if run_tag:
            bundle=load_bundle(PROJECT_ROOT/config.get('parsed_dir','parsed'),project_output,run_tag)
        else:
            st.info('No saved runs found.')
            bundle={'ranked':[],'episodes':[],'validated':[],'scores':[],'bible':{},'exclusions':{},'output':DEMO_ROOT/'unloaded','tag':'unloaded'}
    scope=f"{mode}:{bundle['tag']}"
    if st.session_state.get('decision_scope') != scope:
        st.session_state['decisions']={}
        st.session_state['decision_scope']=scope
        st.session_state.pop('run_message', None)
    bundle['decision_scope']=scope
    overrides={'candidate_provider':cp,'candidate_model':cm,'judge_provider':jp,'judge_model':jm,
               'batch_size':int(batch),'max_api_calls':int(calls),'episodes':selector,'hosted_keys':hosted}
    plan_signature=hashlib.sha256(json.dumps([st.session_state.get('upload_signature'),overrides],sort_keys=True).encode()).hexdigest()
    if estimate_clicked:
        if mode=='Demo':
            st.session_state['estimate']={'bible':{'requests':0,'tokens':0},'candidate_requests':0,'candidate_tokens':0,'judge_requests':0,'judge_tokens':0}
            st.session_state['plan_signature']=plan_signature
        elif ready:
            try:
                st.session_state['estimate']=run_pipeline(active_config,input_dir,upload_tag,overrides|{'dry_run':True})
                st.session_state['plan_signature']=plan_signature
            except Exception as error:
                st.error(redact(error,keys))
    default=st.query_params.get('view','Run')
    tabs=st.tabs(TABS,default=default if default in TABS else 'Run')
    with tabs[0]:
        st.html(render_html('<div class="eyebrow">FROM SCRIPT TO SHORTLIST</div>'))
        st.subheader('A stronger first frame. A harder last question.')
        if mode=='Demo':
            st.caption('THE MIDNIGHT PATENT · Original fictional demo · Deterministic MockProvider scores · No live AI calls')
        episodes=bundle['episodes']
        metrics=st.columns(4)
        for col,label,value in zip(metrics,['Episodes','Total runtime','Ranked promos','Parser warnings'],[len(episodes),format_timestamp(sum(e.duration for e in episodes)),len(bundle['ranked']),sum(len(e.warnings) for e in episodes)]):
            col.metric(label,value)
        if episodes:
            with st.expander('Episode loglines and parser warnings'):
                for ep in episodes:
                    st.write(f'Episode {ep.number}: {ep.logline}')
                    for warning in ep.warnings:
                        st.warning(warning)
        estimate=st.session_state.get('estimate') if not existing and st.session_state.get('plan_signature')==plan_signature else None
        if estimate:
            show_estimate(estimate)
        elif not existing:
            st.caption('Estimate cost in the sidebar before starting a live run. Demo results are already prepared.')
        if st.button('Find promos',type='primary',disabled=existing or not ready or (mode!='Demo' and estimate is None),width='stretch'):
            if mode=='Demo':
                st.success('Demo ready: 11 precomputed promos. Open Top promos to review them.')
                for stage in STAGES:
                    st.status(stage.title()+' · precomputed',state='complete')
            else:
                statuses={}
                def progress(event):
                    stage=event['stage']
                    if stage not in statuses:
                        statuses[stage]=st.status(stage.title(),expanded=True)
                    status=statuses[stage]
                    if event['state']=='complete':
                        status.update(label=stage.title()+' complete',state='complete',expanded=False)
                    elif event['state']=='stopped':
                        status.update(label=stage.title()+' stopped · progress saved',state='error')
                    elif event.get('episode'):
                        status.write(f"Episode {event['episode']} · {event.get('done','')} / {event.get('total','')}")
                try:
                    result=run_pipeline(active_config,input_dir,upload_tag,overrides,progress)
                    if result.get('stopped'):
                        stats=result.get('stats',result.get('judge_stats',{}))
                        message='Daily quota reached — progress saved, resume later.' if stats.get('stopped_daily_quota') or stats.get('stopped')=='LLMQuotaError' else 'Run stopped — progress saved. Review the call budget or connection, then resume.'
                        st.session_state['run_message']=message
                    else:
                        st.session_state['run_message']='Promo shortlist ready.'
                    st.rerun()
                except Exception as error:
                    for status in statuses.values():
                        status.update(label='Run stopped · progress saved',state='error')
                    message=redact(error,keys)
                    if 'daily' in message.lower() and 'quota' in message.lower():
                        message='Daily quota reached — progress saved, resume later.'
                    st.error(message)
        if mode=='Run on my files' and st.session_state.get('run_message'):
            st.info(st.session_state['run_message'])
        if bundle['ranked'] or bundle.get('trailers'):
            show_downloads(bundle,st.session_state['decisions'],'run')
        st.markdown('**The editorial flow**  ·  Parse → Context → Find → Validate → Judge → Rank')
        st.caption('AI proposes and scores. Code checks complete shots and boundaries. You decide what gets posted.')
    with tabs[1]:
        show_promos(bundle)
        if bundle['ranked'] or bundle.get('trailers'):
            show_downloads(bundle,st.session_state['decisions'],'top')
    with tabs[2]:
        all_candidates(bundle)
    with tabs[3]:
        show_bible(bundle)
    with tabs[4]:
        show_timeline(bundle)
    with tabs[5]:
        show_quality(bundle, active_config or config)
    with tabs[6]:
        show_guide(config, expanded=st.query_params.get('guide')=='expanded')
