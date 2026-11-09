"""Streamlit presentation; all execution delegates to the shared pipeline."""

import hashlib
import json
import os
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import dotenv_values

from ..cliprender import render_clip
from ..pipeline import parse_inputs, run_pipeline
from ..llm import CallBudget, LLMRequestError, get_llm, step_model_config
from ..run_status import KEY_NAMES, UNKNOWN_ERROR, describe_failure, error_details
from ..safety_review import shortlist
from ..timeutils import format_timestamp
from .data import DEMO_ROOT, PROJECT_ROOT, existing_run_tags, load_bundle, load_demo, local_runs_allowed, upload_config
from .helpers import filter_promos, redact, render_html, segment_text
from .session import clear_workspace, new_workspace, resolve_credentials, unpack_uploads, upload_run_fingerprint, upload_run_root, write_uploads
from .theme import CSS, card_html
from .quality import show_quality
from .trailers_view import show_trailers
from .downloads import show_downloads
from .guide import show_guide
from .results_upload import MODE as RESULTS_MODE, load_results_zip

TABS=['Run','Top promos','All candidates','Story bible','Timeline','Quality','How it works']
STAGES=['parse','bible','candidates','validate','judge','rank','trailers','editor']


def reset_session():
    epoch=st.session_state.get('ui_epoch',0)+1
    if st.session_state.get('workspace'):
        clear_workspace(st.session_state['workspace'])
    if st.session_state.get('results_workspace'):
        clear_workspace(st.session_state['results_workspace'])
    st.session_state.clear()
    st.session_state['ui_epoch']=epoch


def model_controls(config,step,label,available,public):
    epoch=st.session_state.get('ui_epoch',0)
    current=config['models'][step]
    providers=list(dict.fromkeys([current['provider'],'gemini','groq','openai','ollama','mock']))
    if public:
        providers = [provider for provider in providers if provider != 'ollama']
    def provider_label(name):
        if name == 'mock':
            return 'mock — test run, no API key, fake scores'
        return f'{name} — needs your API key' if name in KEY_NAMES and name not in available else name
    provider=st.selectbox(f'{label} provider',providers,key=f'{step}_provider_{epoch}',
                          format_func=provider_label)
    models=list(dict.fromkeys([v['model'] for v in config['models'].values() if v['provider']==provider]))
    if not models:
        models=[{'mock':'mock','groq':'openai/gpt-oss-120b','gemini':'gemini-2.5-flash'}.get(provider,'custom-model')]
    selected=st.selectbox(f'{label} model',models,key=f'{step}_model_choice_{provider}_{epoch}')
    model=st.text_input(f'Edit {label.lower()} model',selected,key=f'{step}_model_{provider}_{selected}_{epoch}')
    return provider,model


def show_run_failure(failure):
    with st.status(f"{failure['stage'].title()} stopped · {failure['kind']}", expanded=True, state='error'):
        st.text(f"{failure['provider']} / {failure['model']}")
        st.write(failure.get('message') or UNKNOWN_ERROR)
        for detail in failure.get('errors', []):
            episode = f"Episode {detail['episode']} · " if detail['episode'] is not None else ""
            http = f" · HTTP {detail['status']}" if detail['status'] else ""
            st.text(f"{episode}{failure['provider']} / {failure['model']} · {detail['error_class']}{http}\n"
                    f"{detail['message'] or UNKNOWN_ERROR}")
        st.write(f"Episodes done: {', '.join(map(str, failure['done'])) or 'none'} · "
                 f"Pending: {', '.join(map(str, failure['pending'])) or 'none'}")
        st.write(f"Calls: {failure['calls']} · Includes {failure['retries']} budgeted retries")
        st.write(failure['next_step'])


def test_selected_keys(config, keys, providers):
    """One uncached attempt per distinct provider, using only a synthetic prompt."""
    with new_workspace() as workspace:
        probe = upload_config(config, Path(workspace), keys)
        probe['_interactive_run'] = True
        probe['llm'].update(max_retries=1, use_cache=False)
        for provider, model in providers.items():
            key = KEY_NAMES.get(provider)
            if key and not keys.get(key):
                yield provider, False, f'{key} is missing. Paste a key in API keys.'
                continue
            try:
                client = get_llm(probe, provider, model)
                client.call_budget = CallBudget(1)
                if provider == 'gemini':
                    client.thinking_budget = 0
                elif provider in {'groq', 'openai'}:
                    client.max_completion_tokens = 128
                client.generate_text('Reply with OK.', 'Reply with OK.', 'test_api_key')
            except (LLMRequestError, ValueError) as error:
                yield provider, False, describe_failure(error)[2]
            else:
                yield provider, True, 'Key accepted · 1 call.'


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
    hosted_cap = config.get('hosted', {}).get('max_api_calls', 30)
    epoch=st.session_state.setdefault('ui_epoch',0)
    project_output=PROJECT_ROOT/config.get('output_dir','outputs')
    modes=['Demo','Run on my files',RESULTS_MODE]
    if allow_existing:
        modes.append('Open existing run')
    with st.sidebar:
        st.html(render_html('<div class="eyebrow">KUKU / PROMO STUDIO</div>'))
        mode=st.radio('Mode',modes,key=f'mode_{epoch}')
        existing=allow_existing and mode=='Open existing run'
        viewing_results = mode == RESULTS_MODE
        read_only = existing or viewing_results
        with st.expander('Reviewer? Start here'):
            st.markdown('''- **Load results bundle (.zip)** → upload `raw_outputs.zip` from the private submission link.
- **Demo** → browse fictional results without API keys.
- **Run on my files** → paste keys → **Test API keys** → **Find promos**.''')
        results_zip = st.file_uploader('Results ZIP', type=['zip'], max_upload_size=50,
                                       key=f'results_zip_{epoch}') if viewing_results else None
        run_tag=None
        if existing:
            tags=existing_run_tags(project_output)
            if tags:
                run_tag=st.selectbox('Saved run',tags,index=tags.index('gemini_full') if 'gemini_full' in tags else 0)
            st.caption('Local results are opened read-only.')
        st.caption('Demo is fictional, precomputed and key-free.' if mode=='Demo' else 'Uploads and outputs stay in this session’s temporary folder.' if not existing else 'Browse saved results on this computer.')
        uploads=st.file_uploader('Input pack · TXT or ZIP',type=['txt','zip'],accept_multiple_files=True,disabled=mode!='Run on my files',key=f'uploads_{epoch}')
        secrets, environment, pasted = {}, {}, {}
        if mode == 'Run on my files':
            try:
                secrets=st.secrets.to_dict()
            except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):
                pass
            environment = (dotenv_values(PROJECT_ROOT/'.env') if allow_existing else {}) | dict(os.environ)
        with st.expander('API keys · optional'):
            for name in KEY_NAMES.values():
                pasted[name]=st.text_input(name,type='password',key=f'{name}_{epoch}',
                                           disabled=mode!='Run on my files').strip()
            st.caption('Pasted keys override app keys for this session. They are never logged or saved. Hosted caps are lifted only when every active provider that needs a key uses your own key.')
        available_keys, _ = resolve_credentials(pasted,secrets,environment,KEY_NAMES)
        available = {provider for provider,key in KEY_NAMES.items() if key in available_keys}
        with st.expander('Models',expanded=mode=='Run on my files'):
            cp,cm=model_controls(config,'candidates','Candidate',available,not allow_existing)
            jp,jm=model_controls(config,'judge','Judge',available,not allow_existing)
            if mode=='Run on my files':
                for stage in ('trailers','editor'):
                    if config.get(stage,{}).get('enabled',False):
                        setting=config['models'][stage]
                        st.caption(f"Configured {stage}: {setting['provider']} / {setting['model']}")
        batch=st.number_input('Batch size',min_value=1,max_value=20,value=5,key=f'batch_{epoch}')
        calls=st.number_input('Max API calls',min_value=0,max_value=1000,value=hosted_cap,key=f'calls_{epoch}')
        selector=st.text_input('Episodes','all',help='Examples: all, 1-10, 1,3,5',key=f'selector_{epoch}')
        estimate_clicked=st.button('Estimate cost',width='stretch',disabled=read_only)
        st.button('Clear session',width='stretch',on_click=reset_session)
    st.html(render_html('<h1 class="royal-title">Kuku Promo Finder</h1><div class="royal-subtitle">Find the moment that makes them watch the next episode.</div>'))
    keys={}
    hosted=False
    bundle=load_demo()
    active_config=None
    input_dir=None
    upload_key = None
    run_info = None
    ready=mode=='Demo'
    mock_upload=mode=='Run on my files' and 'mock' in (cp,jp)
    upload_tag='session_mock' if mock_upload else 'session'
    if mock_upload:
        cp=jp=cm=jm='mock'
    if mode=='Run on my files':
        bundle={'ranked':[],'episodes':[],'validated':[],'scores':[],'bible':{},'exclusions':{},'output':DEMO_ROOT/'unloaded','tag':'session'}
        providers={cp,jp,config['models']['bible']['provider']}
        providers.update(config['models'][stage]['provider'] for stage in ('trailers','editor') if config.get(stage,{}).get('enabled',False))
        keys,hosted=resolve_credentials(pasted,secrets,environment,providers,environment_is_hosted=not allow_existing)
        if mock_upload:
            keys,hosted={},False
            st.info('Mock upload run. All stages use test responses with fake scores.')
        test_keys = st.sidebar.button('Test API keys', disabled=mock_upload, width='stretch')
        st.sidebar.caption('Each test with a key makes one provider call, separate from Find promos.')
        if test_keys:
            selected = [(cp, cm), (jp, jm)]
            selected.extend(step_model_config(config, stage) for stage in ('bible','trailers','editor')
                            if stage == 'bible' or config.get(stage, {}).get('enabled'))
            by_provider = {}
            for provider, model in selected:
                if provider in KEY_NAMES:
                    by_provider.setdefault(provider, model)
            for provider, ok, message in test_selected_keys(config, keys, by_provider):
                if ok:
                    st.sidebar.success(f'✓ {provider} · {message}')
                else:
                    st.sidebar.error(f'✗ {provider} · {message}')
        if hosted:
            calls=min(hosted_cap,calls)
            st.sidebar.caption(f'Hosted-key limits: 20 episodes and {hosted_cap} API calls per attempt.')
        if cp == 'groq':
            batch = 1
            st.sidebar.caption('Groq candidate batches contain one episode.')
        if uploads:
            try:
                files=unpack_uploads([(file.name,file.getvalue()) for file in uploads])
                models = {stage: step_model_config(config, stage) for stage in ('bible','trailers','editor')
                          if stage == 'bible' or config.get(stage,{}).get('enabled')}
                models.update(candidates=(cp,cm),judge=(jp,jm))
                if mock_upload:
                    models = {stage: ('mock','mock') for stage in models}
                fingerprint = upload_run_fingerprint(files, models, selector, PROJECT_ROOT/'src/prompts')
                generations = st.session_state.setdefault('run_generations',{})
                generation = generations.get(fingerprint,0)
                if not st.session_state.get('workspace'):
                    st.session_state['workspace']=new_workspace()
                root=upload_run_root(st.session_state['workspace'],fingerprint,generation)
                upload_key = root.name
                input_dir=write_uploads(root,files)
                active_config=upload_config(config,root,keys,mock_upload)
                preview=parse_inputs(input_dir,root/'parsed',dry_run=True)
                if hosted and len(preview)>20:
                    raise ValueError(f'Hosted runs are capped at 20 episodes / {hosted_cap} calls. Paste your own API keys or use the CLI for a full show.')
                bundle=load_bundle(root/'parsed',root/'outputs',upload_tag)
                bundle['episodes']=preview
                run_info = st.session_state.setdefault('upload_runs',{}).get(upload_key)
                ready=True
            except (OSError,ValueError) as error:
                message = redact(error,keys) if isinstance(error,ValueError) else 'Could not read the uploaded episodes.'
                st.error(message)
        else:
            st.info('Upload your episode TXT files or a ZIP input pack to begin.')
    if existing:
        if run_tag:
            bundle=load_bundle(PROJECT_ROOT/config.get('parsed_dir','parsed'),project_output,run_tag)
        else:
            st.info('No saved runs found.')
            bundle={'ranked':[],'episodes':[],'validated':[],'scores':[],'bible':{},'exclusions':{},'output':DEMO_ROOT/'unloaded','tag':'unloaded'}
    results_digest = None
    if viewing_results:
        bundle = {'ranked':[], 'episodes':[], 'validated':[], 'scores':[], 'bible':{}, 'exclusions':{},
                  'output':DEMO_ROOT/'unloaded', 'tag':'unloaded'}
        results_digest = hashlib.sha256(results_zip.getbuffer()).hexdigest() if results_zip else None
        if st.session_state.get('results_digest') != results_digest:
            old_workspace = st.session_state.pop('results_workspace', None)
            if old_workspace:
                clear_workspace(old_workspace)
            st.session_state.pop('results_bundle', None)
            st.session_state.pop('results_error', None)
            st.session_state['results_digest'] = results_digest
            if results_zip:
                workspace = new_workspace()
                try:
                    st.session_state['results_bundle'] = load_results_zip(results_zip.getvalue(), workspace.name)
                    st.session_state['results_workspace'] = workspace
                except (OSError, ValueError) as error:
                    clear_workspace(workspace)
                    st.session_state['results_error'] = str(error)
        if st.session_state.get('results_bundle'):
            bundle = dict(st.session_state['results_bundle'])
            st.info(f"Viewing saved results for {bundle['tag']} · read-only.")
        elif st.session_state.get('results_error'):
            st.error(st.session_state['results_error'])
        else:
            st.info('Upload a results ZIP to browse a saved run.')
    scope=f"{mode}:{bundle['tag']}:{results_digest or upload_key}"
    if st.session_state.get('decision_scope') != scope:
        st.session_state['decisions']={}
        st.session_state['decision_scope']=scope
    bundle['decision_scope']=scope
    overrides={'candidate_provider':cp,'candidate_model':cm,'judge_provider':jp,'judge_model':jm,
               'batch_size':int(batch),'max_api_calls':int(calls),'episodes':selector,'hosted_keys':hosted,'interactive':True}
    plan_signature=hashlib.sha256(json.dumps([upload_key,overrides],sort_keys=True).encode()).hexdigest()
    if estimate_clicked:
        if mode=='Demo':
            st.session_state['estimate']={'bible':{'requests':0,'tokens':0},'candidate_requests':0,'candidate_tokens':0,'judge_requests':0,'judge_tokens':0}
            st.session_state['plan_signature']=plan_signature
        elif ready:
            try:
                st.session_state['estimate']=run_pipeline(active_config,input_dir,upload_tag,overrides|{'dry_run':True})
                st.session_state['plan_signature']=plan_signature
            except Exception as error:
                st.error('Could not estimate this upload. Check the episode selection and model settings.')
    if run_info and bundle['ranked']:
        st.info(f"Results from: candidates {cp}/{cm}, judge {jp}/{jm} · "
                f"run started {run_info['started']} · resumed {'yes' if run_info['resumed'] else 'no'}")
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
        estimate=st.session_state.get('estimate') if not read_only and st.session_state.get('plan_signature')==plan_signature else None
        if estimate:
            show_estimate(estimate)
        elif not read_only:
            st.caption('Use Estimate cost in the sidebar to preview a live run. Demo results are already prepared.')
        run_buttons = st.columns(2) if mode == 'Run on my files' else (st,)
        find = run_buttons[0].button('Find promos',type='primary',disabled=read_only or not ready,width='stretch')
        if mode == 'Run on my files' and run_buttons[1].button('Start fresh run', disabled=not ready, width='stretch'):
            generations[fingerprint] = generation + 1
            st.session_state['launch_fresh'] = True
            st.rerun()
        launch_requested = (mode == 'Run on my files' and ready and
                            (st.session_state.pop('retry_with_groq', False) or st.session_state.pop('launch_fresh', False)))
        if find or launch_requested:
            if mode=='Demo':
                st.success('Demo ready: 11 precomputed promos. Open Top promos to review them.')
                for stage in STAGES:
                    st.status(stage.title()+' · precomputed',state='complete')
            else:
                progress_box = st.empty()
                def progress(event):
                    stage = event['stage'].title()
                    if event['state'] == 'waiting':
                        progress_box.info(f"Waiting for {event['provider']} rate limit… {int(event['seconds'] + .999)}s")
                    elif event['state'] == 'running':
                        progress_box.info(f'{stage} running…')
                    elif event['state'] == 'progress' and event.get('episode'):
                        progress_box.info(f"{stage} · episode {event['episode']} · {event.get('done','')} / {event.get('total','')}")
                runs = st.session_state['upload_runs']
                had_progress = bool(run_info and run_info.get('had_progress'))
                info = {'started': run_info['started'] if had_progress else datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z'),
                        'resumed': had_progress, 'had_progress': had_progress}
                try:
                    result=run_pipeline(active_config,input_dir,upload_tag,overrides,progress)
                    info['failure'] = result.get('failure')
                    info['had_progress'] = had_progress or not info['failure'] or info['failure']['kind'] != 'Missing API key'
                    info['complete'] = not result.get('stopped')
                except (ValueError, OSError):
                    info['failure'] = dict(stage='input',provider=cp,model=cm,kind='Input error',
                                           message='The uploaded episodes or episode selection could not be processed.',
                                           done=[],pending=[ep.number for ep in bundle['episodes']],calls=0,retries=0,
                                           next_step='Check the episode selection and uploaded TXT files, then try again.')
                runs[upload_key] = info
                st.rerun()
        if run_info and run_info.get('failure'):
            failure = run_info['failure']
            show_run_failure(failure)
            if failure['stage'] == 'candidates' and failure['provider'] not in ('groq','mock'):
                def retry_groq():
                    st.session_state[f'candidates_provider_{epoch}'] = 'groq'
                    st.session_state['retry_with_groq'] = True
                st.button('Retry with Groq',on_click=retry_groq)
        elif run_info and run_info.get('complete'):
            st.success('Promo shortlist ready.')
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
