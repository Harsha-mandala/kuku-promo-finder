"""Formatting and editorial helpers with no provider dependencies."""

import csv
import io
from html import escape

from ..timeutils import format_timestamp


def segment_text(row):
    if row.get('segment_text'):
        return row['segment_text'].replace('-', '–')
    return ' · '.join(f"E{s['episode']:02d} {format_timestamp(s['start'])}–{format_timestamp(s['end'])}" for s in row.get('segments', []))


def filter_promos(rows, episode_range, min_score=0, hide_caution=False, statuses=None):
    return [r for r in rows if episode_range[0] <= r['episode'] <= episode_range[1]
            and r['final_score'] >= min_score
            and (not hide_caution or r['ad_safety'] != 'caution')
            and (statuses is None or r['status'] in statuses)]


def decisions_csv(decisions):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['promo_id','decision','note'])
    writer.writeheader()
    for key, value in sorted(decisions.items()):
        note = value.get('note','')
        if note.startswith(('=','+','-','@')):
            note = "'" + note
        writer.writerow({'promo_id':key,'decision':value['decision'],'note':note})
    return stream.getvalue().encode('utf-8')


def redact(message, keys):
    result = str(message)
    for key in keys.values():
        if key:
            result = result.replace(key, '[redacted]')
    return result


def score_bars(row):
    bars=[]
    for label, field, maximum in [('Hook','hook_score',5),('Cliffhanger','cliffhanger_score',5),('Standalone','standalone_score',5),('Emotion','emotion_score',5),('Post priority','post_priority',10)]:
        value=row.get(field)
        text=f'{value}/{maximum}' if value is not None else 'not scored'
        width=100*value/maximum if value is not None else 0
        bars.append(f'<div class="score-label">{label}<span>{text}</span></div><div class="score-track"><i style="width:{width:.1f}%"></i></div>')
    return ''.join(bars)


def safety_chip(row):
    rating=row.get('ad_safety','safe')
    verified=row.get('ad_safety_verified',rating=='safe')
    result=f'<span class="chip {escape(rating)}">{escape(rating)}</span>'
    if not verified:
        result+='<span class="chip unverified">unverified</span>'
    return result
