"""Trailer cards and ordered segment controls using saved results only."""

from html import escape

import streamlit as st

from ..timeutils import format_timestamp
from ..trailers import ValidatedTrailer, render_trailer
from .theme import card_html


def show_trailers(bundle,episodes):
    rows = bundle.get('trailers',[])
    if not rows:
        return
    st.subheader('Trailers')
    validated = {r['id']:ValidatedTrailer.model_validate(r) for r in bundle.get('trailer_validated',[])}
    for row in rows:
        st.markdown(card_html(row),unsafe_allow_html=True)
        chips = []
        for index,segment in enumerate(row['segments'],1):
            label = f"E{segment['episode']:02d} {format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}"
            chips.append(f'<span class="chip status">{index}. {escape(label)}</span><button class="copy-segment" data-segment="{escape(label,quote=True)}" aria-label="Copy trailer segment {index}">Copy {index}</button>')
        st.markdown('<div class="segment-chips">'+' <span>→</span> '.join(chips)+'</div>',unsafe_allow_html=True)
        st.caption('Story: '+row['story']+f" · coherence {row['coherence']}/5 · spoiler risk {row['spoiler_risk']}")
        with st.expander('Trailer script · CUT markers',expanded=st.query_params.get('expand_trailer')=='1'):
            if row['promo_id'] in validated:
                st.code(render_trailer(validated[row['promo_id']],episodes),language=None)
    st.subheader('Singles')
