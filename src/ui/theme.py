"""Embers presentation with escaped source text."""

from html import escape

from .helpers import safety_chip,score_bars,segment_text

CSS='''<style>
.block-container{max-width:1240px;padding-top:2.2rem;padding-bottom:3rem}
.royal-title{font-size:2.9rem;font-weight:800;color:#FE9677;letter-spacing:-1.5px;display:inline-block;border-bottom:2px solid #984063;padding-bottom:8px;margin:0}
h1.royal-title{background:linear-gradient(90deg,#FE9677,#F64668);background-clip:text;-webkit-background-clip:text;color:transparent!important;font-size:2.9rem!important;font-weight:800!important}
.royal-subtitle{color:#C8BFD6;margin:12px 0 28px;font-size:1.05rem}
.eyebrow{color:#C8BFD6;text-transform:uppercase;font-size:.75rem;letter-spacing:2px;font-weight:700}
.promo-card{background:#2A2946;border:1px solid #41436A;border-left:5px solid var(--accent);border-radius:14px;padding:24px;margin:18px 0 4px}
.card-top>div{flex:1;min-width:0}.card-top .meta{display:flex;justify-content:space-between;gap:18px;align-items:center}
.card-top{display:flex;align-items:center;gap:14px;margin-bottom:18px}
.rank-badge{border-radius:50%;background:#FE9677;color:#1C1B31;min-width:40px;height:40px;display:inline-flex;align-items:center;justify-content:center;font-weight:800;font-size:1.15rem}
.card-title{font-size:1.2rem;font-weight:700;color:#FFF4EE}.meta{color:#C8BFD6;font-size:.85rem}
.chip{display:inline-block;border-radius:20px;padding:3px 10px;margin:0 5px 5px 0;font-size:.75rem;font-weight:750;color:#1C1B31}
.chip.safe{background:#6FD6B8}.chip.caution{background:#FFC870}.chip.unsafe{background:#FF4D4D}.chip.status{background:#984063;color:#FFF4EE}.chip.unverified{border:1px solid #C8BFD6;color:#C8BFD6;background:transparent}
.quote{padding:9px 14px;border-left:3px solid #F64668;margin:16px 0;color:#FFF4EE;font-size:1.08rem}.quote.cliff{border-left-color:#FE9677}
.quote-label{font-size:.68rem;letter-spacing:1.7px;color:#C8BFD6;display:block;margin-bottom:6px}
.score-label{color:#FFF4EE;font-size:.75rem;margin:8px 0 5px}.score-label span{float:right;color:#C8BFD6}
.score-track{height:4px;background:#41436A;border-radius:4px;overflow:hidden}.score-track i{display:block;height:4px;background:linear-gradient(90deg,#984063,#F64668,#FE9677)}
.script-line{padding:7px 12px;font-size:.84rem;line-height:1.6;border-left:3px solid transparent}.script-line.hook{border-left-color:#F64668;background:#2A2946}.script-line.cliff{border-left-color:#FE9677;background:#2A2946}.script-line.both{border-left-color:#F64668;border-right:3px solid #FE9677}.script-label{color:#C8BFD6;font-size:.68rem;font-weight:700;letter-spacing:1px}
.timeline-row{background:#2A2946;border:1px solid #41436A;border-radius:14px;padding:18px 22px;margin:12px 0}.timeline-track{position:relative;background:#1C1B31;height:38px;border-radius:8px;margin-top:14px;overflow:hidden}
.timeline-span{position:absolute;height:100%;background:#F64668;color:#FFF;display:flex;align-items:center;justify-content:center;font-weight:750;font-size:.76rem;border:2px solid #1C1B31;box-sizing:border-box}.timeline-span.gold{background:#FE9677;color:#1C1B31}
.legend{display:flex;gap:20px;color:#C8BFD6;font-size:.8rem;margin:16px 0}.section-intro{color:#C8BFD6;font-size:.95rem;margin-bottom:18px}
div[data-testid="stMetric"]{background:#2A2946;border:1px solid #41436A;padding:16px 20px;border-radius:14px}
button[kind="primary"]{font-weight:750;font-size:1.1rem}button[kind="primary"]:hover{border-color:#FE9677}
[data-testid="stSidebar"]{border-right:1px solid #41436A}[data-testid="stTabs"] button{color:#FFF4EE}
[role="tab"]:focus,[role="tab"]:focus-visible{outline:none!important;box-shadow:none!important}[role="tab"][aria-selected="true"]{color:#F64668!important;border-bottom:3px solid #F64668!important}
[data-testid="stAlert"]{background:#2A2946;border-left:4px solid #FE9677;color:#FFF4EE}
[data-testid="stAlert"] :is(p,li,span,a,strong,em,code){color:#FFF4EE!important}
button[kind="primary"],button[kind="primary"] p{color:#1C1B31!important}
.copy-segment{float:right;border:1px solid #C8BFD6;border-radius:8px;background:#2A2946;color:#FFF4EE;padding:5px 10px;cursor:pointer}
.timeline-span.trailer-span{z-index:2;top:2px;height:calc(100% - 4px);align-items:flex-end;background:transparent;color:#FFF4EE;border:2px dashed #FFF4EE;border-radius:5px;text-shadow:0 1px 3px #1C1B31,0 0 5px #1C1B31;font-size:.67rem}
.segment-chips{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:12px 0}.segment-chips .copy-segment{float:none}
</style>'''


def card_html(row):
    rank=row['rank']
    accent='#FE9677' if rank<=3 else '#F64668' if rank<=10 else '#984063'
    title='Trailer' if row.get('type')=='trailer' else f"Episode {row['episode']}"
    ranks=''
    if row.get('editor_rank') is not None:
        ranks=f"<div class='meta'>Judge #{row['judge_rank']} · Editor #{row['editor_rank']} · Final #{row['final_rank']}</div>"
        if row.get('rank_disagreement'):
            ranks+="<span class='chip status'>Rank disagreement ≥5</span>"
    elif row.get('final_rank') is not None:
        ranks=f"<div class='meta'>Judge #{row['judge_rank']} · Editor: not reviewed · Final #{row['final_rank']}</div>"
    return f'''<div class="promo-card" style="--accent:{accent}">
    <div class="card-top"><span class="rank-badge">{rank}</span><div><div class="card-title">{title} · {escape(row['promo_id'])}</div><div class="meta">{escape(segment_text(row))} · {row['duration_s']:.1f} s · score {row.get('score_adj',row['final_score']):.2f} <button class="copy-segment" data-segment="{escape(segment_text(row),quote=True)}" aria-label="Copy segment timestamp">Copy</button></div>{ranks}</div></div>
    {safety_chip(row)}<span class="chip status">{escape(row['status'])}</span>
    {"<span class='chip caution'>Needs safety review</span>" if row.get('safety_review') else ""}
    <div class="quote"><span class="quote-label">THE HOOK</span>{escape(row['hook_text'])}</div>
    <div class="quote cliff"><span class="quote-label">THE CLIFFHANGER</span>{escape(row['cliffhanger_text'])}</div>
    <p>{escape(row.get('why_it_works') or '')}</p><div class="meta">{escape(row.get('why_post_first') or row.get('one_line_pitch',''))}</div>
    {score_bars(row)}</div>'''
