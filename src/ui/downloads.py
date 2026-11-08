"""Build editor exports in memory for saved runs and temporary uploads."""

import csv
import io
import json
import zipfile

import streamlit as st

from ..timeutils import format_timestamp
from .helpers import decisions_csv


SEGMENT_FIELDS = ('final_rank', 'promo_id', 'type', 'segment_no', 'episode',
                  'start', 'end', 'duration', 'hook', 'cliffhanger', 'why_post_first')


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue().encode('utf-8')


def posting_rows(bundle):
    singles = [dict(row, type='single') for row in bundle['ranked']]
    trailers = [dict(row, type='trailer') for row in bundle.get('trailers', [])]
    rows = singles + trailers
    if bundle.get('editor_order'):
        rows.sort(key=lambda row: row['final_rank'])
    else:
        for rank, row in enumerate(rows, 1):
            row['final_rank'] = rank
    return rows


def editor_ready_csv(rows):
    segments = []
    for row in rows:
        if row.get('ad_safety') == 'unsafe':
            continue
        for number, segment in enumerate(row['segments'], 1):
            start, end = segment['start'], segment['end']
            segments.append(dict(final_rank=row['final_rank'], promo_id=row['promo_id'], type=row['type'],
                                 segment_no=number, episode=segment['episode'], start=format_timestamp(start),
                                 end=format_timestamp(end), duration=round(end-start, 3), hook=row['hook_text'],
                                 cliffhanger=row['cliffhanger_text'], why_post_first=row.get('why_post_first', '')))
    return csv_bytes(segments, SEGMENT_FIELDS)


def download_files(bundle, decisions):
    rows = posting_rows(bundle)
    files = {}
    for name, kind in [('promos_ranked', 'single'), ('trailers_ranked', 'trailer'), ('promos_all', None)]:
        selected = [row for row in rows if kind is None or row['type'] == kind]
        fields = list(dict.fromkeys(key for row in selected for key in row)) or ['promo_id', 'type', 'segments']
        files[name+'.csv'] = csv_bytes(selected, fields)
        if kind:
            files[name+'.json'] = json.dumps(selected, ensure_ascii=False, indent=2).encode('utf-8')
    files['editor_ready.csv'] = editor_ready_csv(rows)
    files['editor_decisions.csv'] = decisions_csv(decisions)
    output, tag = bundle['output'], bundle['tag']
    for name, path in [('report.md', output/f'final_{tag}/report.md'),
                       ('story_bible.json', output/'story_bible.json'),
                       ('eval_report.md', output/f'eval_{tag}/eval_report.md')]:
        if path.is_file():
            files[name] = path.read_bytes()
    return files


def download_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return stream.getvalue()


def show_downloads(bundle, decisions, location):
    st.subheader('Downloads')
    if not bundle.get('editor_order'):
        st.caption('The export uses judge order, with singles first, until editor ranks are available.')
    files = download_files(bundle, decisions)
    labels = {'promos_ranked.csv':'Ranked singles CSV', 'promos_ranked.json':'Ranked singles JSON',
              'trailers_ranked.csv':'Trailers CSV', 'trailers_ranked.json':'Trailers JSON',
              'promos_all.csv':'Combined promos CSV', 'editor_ready.csv':'Editor-ready CSV',
              'editor_decisions.csv':'Editor decisions CSV', 'report.md':'Run report'}
    columns = st.columns(4)
    for index, (name, label) in enumerate(labels.items()):
        if name in files:
            mime = 'application/json' if name.endswith('.json') else 'text/csv' if name.endswith('.csv') else 'text/markdown'
            columns[index % 4].download_button(label, files[name], file_name=name, mime=mime,
                                               key=f'{location}_{name}', width='stretch')
    st.download_button('Download everything', download_zip(files), file_name=f"{bundle['tag']}_editor_pack.zip",
                       mime='application/zip', key=f'{location}_zip')
