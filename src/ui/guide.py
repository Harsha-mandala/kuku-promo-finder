"""In-app help for the controls and scores shown by the current configuration."""

import streamlit as st

from ..rank import DEFAULT_PENALTIES, DEFAULT_WEIGHTS


def show_guide(config, expanded=False):
    st.subheader('How it works')
    rank = config.get('rank', {})
    weights = DEFAULT_WEIGHTS | rank.get('weights', {})
    penalties = DEFAULT_PENALTIES | rank.get('penalties', {})
    editor = config.get('editor', {})
    diversity = rank.get('same_episode_penalty', .05)

    with st.expander('1. Overview', expanded=expanded):
        st.markdown('''Kuku Promo Finder helps promo editors turn screenplay packs into a shortlist of timecoded clips for review and cutting.

**Parse → Story bible → Find candidates → Validate → Judge → Rank → Trailers → Editor pass → Human review**

- **Parse** uses code to read episode text, assign shot and line IDs, and resolve screenplay timing.
- **Story bible** uses AI to summarize the show and tie character aliases to source evidence.
- **Find candidates** uses AI to propose clips by existing IDs. Code measures their spans and drops overlapping or surplus proposals.
- **Validate** uses code to repair whole-shot boundaries and timing, flag issues, and reject clips that fail hard checks.
- **Judge** uses a separate AI prompt to score the clip and give a safety reason. Code checks cited IDs and caps the number of scored singles per episode.
- **Rank** uses code to combine scores, timing, safety and episode diversity.
- **Trailers** uses AI to assemble source segments and score their coherence. Code validates the segments and ranks accepted trailers.
- **Editor pass** uses AI to compare a shuffled shortlist and explain its order. Code blends that order with judge rank.
- **Human review** is where you check footage, cuts and safety, then approve clips for editing.''')

    with st.expander('2. Modes', expanded=expanded):
        st.markdown('''- **Demo** opens four fictional episodes with precomputed mock results and fictional gold labels.
- **Run on my files** uploads a new show into a temporary session. Select mock in either provider menu to run the entire pipeline with fake scores and no key.
- **Load results bundle (.zip)** opens a saved run in the local or deployed app. Upload the submission ZIP to browse all tabs and download results. The app reads the bundle from a temporary session folder. Clear session deletes that folder.
- **Open existing run** appears only in the local app. Choose a saved tag to browse its results and download files. Your notes and decisions live in the current session. Saved run files stay read-only.''')

    with st.expander('3. Tabs and controls', expanded=expanded):
        st.markdown('''**Run** shows episode count, runtime, ranked singles and parser warnings. Expand the loglines and warnings to check ingestion. The sidebar selects the mode, saved tag or upload pack, candidate and judge providers and models, password inputs, batch size, maximum calls and episode scope. The editable model name supports a provider model outside the suggested list. Trailer, editor and bible models come from config. **Estimate cost** previews input tokens and request counts. **Find promos** runs the selected pack after an estimate, or opens the prepared demo. Progress cards show each stage. **Clear session** removes the temporary upload and clears notes and keys. The download panel appears when results are available.

**Top promos** filters on opening episode, minimum score, validation status and whether to hide caution clips. **Safe-only top 10** selects up to ten safe clips within the active filters. **Needs safety review only** opens clips held for human review. Unsafe clips remain excluded from the cards. Read the hook, cliffhanger, safety, scores and ranks on each card. **Copy** copies timecodes. Expand the script and validator notes to inspect the exact lines and fixes. Add an **Editor note**, then use **Approve** or **Reject** to record a session decision. Trailer cards show their segments in play order, with one copy button per segment and a script separated by cut markers. Downloads include the whole active run.

**All candidates** includes pending, rejected, unsafe and ranked clips. Choose which table columns appear, click a heading to sort and scroll horizontally for later columns. Read the fix or rejection reason alongside the status, duration and available scores.

**Story bible** shows the premise, characters, aliases, evidence and arcs. Use the character table to check the context used during generation.

**Timeline** shows episode time horizontally. Coral marks the first three ranks, pink marks other singles and dashed outlines show trailer segments on their source episodes. Hover to read timecodes. Trailer segment numbers show play order across episodes.

**Quality** shows saved hook and IoU evaluation, baseline comparisons, pipeline health, judge score counts, safety, human agreement and cost when available. Expand lenient comparison, allocated episode cost, failure gallery or provider comparison for detail. Download the evaluation report for the recorded metrics.''')

    with st.expander('4. Glossary and scoring', expanded=expanded):
        st.markdown(f'''- **Shot** is a camera unit with source lines attached. Its **effective span** includes attached audio or dialogue that extends beyond the camera timestamps.
- **Clean cut** is a boundary that avoids cutting across a line. `[!start]` and `[!end]` mark unsafe boundaries in source views.
- **Hook** is the opening beat that grabs attention within three seconds. **Cliffhanger** is the unresolved beat at the end.
- **Segment** is one continuous range of complete shots. A **trailer** plays two or three such segments in order.
- **ok** passed timing and boundary checks. **fixed** means code made a repair such as advancing the start to the hook shot. **flagged** means a concern remains, such as an unsafe boundary. **rejected** means a hard check failed, such as a duration outside 20–60 seconds.
- **hook_reassigned** means a different source line became the hook. **cliffhanger_reassigned** means the ending now cites another source beat. **weak_hook_kind** marks a sound or camera-only hook.
- **Hook score** measures opening impact. **Cliffhanger score** measures the ending’s pull. **Standalone** measures how well a new viewer can follow the clip. **Emotion** measures the strength of the emotional beat. Each uses 1–5. **Post priority** uses 1–10 for posting appeal. **Coherence** uses 1–5 for how well a trailer’s segments work together.
- **Judge mix** normalizes each scale to 0–1, then weights hook {weights['hook']:g}, cliffhanger {weights['cliffhanger']:g}, standalone {weights['standalone']:g}, emotion {weights['emotion']:g} and priority {weights['post_priority']:g}. Older scores without priority use the legacy mix shown on the card.
- **Final score** combines judge mix × {weights['judge']:g}, duration and boundary fit × {weights['rule']:g}, and early episode preference × {weights['early_episode']:g}. Subtract {penalties['flagged']:g} for flagged status, {penalties['hook_reassigned']:g} for hook reassignment, {penalties['cliffhanger_reassigned']:g} for cliffhanger reassignment and {penalties['caution']:g} for caution when applicable. Trailers add 0.10 × normalized coherence.
- **Same-episode penalty** subtracts {diversity:g} from the adjusted ranking score per already placed clip from the same opening episode.
- **Safe** is a judge assessment of lower concern. **Caution** calls for review. **Unsafe** is excluded from ranked exports. **Unverified** means the safety reason lacks a verified source citation.
- **Judge rank** is the score-based order. **Editor rank** is the comparison pass order. **Final rank** blends their normalized positions with weights {editor.get('judge_weight', .6):g} and {editor.get('editor_weight', .4):g}. Items outside the editor pool follow the blended shortlist. The disagreement badge marks a difference of five or more ranks.''')

    with st.expander('5. Run a new show', expanded=expanded):
        st.markdown('''1. Prepare one UTF-8 TXT per episode using the screenplay format with an episode number and timestamped camera, dialogue and action blocks. Upload the TXT files or one ZIP of them.
2. Choose providers and models. Get Gemini keys in Google AI Studio and Groq keys in the Groq Console. Paste them into the password fields. Pasted keys are held in the session and never saved to disk. Mock uses fake scores without keys.
3. Set episode scope, batch size and maximum calls, then select **Estimate cost**. The token estimate covers inputs. Output tokens and retry costs depend on responses.
4. Select **Find promos**. If a budget, quota or connection stops a run, keep the same upload and session, adjust the budget or wait for the quota, estimate again and resume. Saved work is reused when inputs match.
5. Hosted runs allow up to 20 episodes and 30 budgeted calls per attempt by default. A per-minute rate-limit response does not use a hosted call. The app waits up to about three minutes, then saves progress so you can resume. Use your own keys for all active stages or run the CLI for a full show.''')
        st.code("$env:PYTHONIOENCODING = 'utf-8'\n.\\.venv\\Scripts\\python.exe run.py --input data --tag my_show --episodes all --batch-size 5 --max-calls-gemini 16 --max-calls-groq 60", language='powershell')
        st.caption('Add --dry-run to preview the CLI plan. Reuse the same tag to resume, or scope a deliberate refresh with --force-episodes.')

    with st.expander('6. Downloads', expanded=expanded):
        st.markdown('''- **Editor-ready CSV** is the cutting handoff. Each row is one segment. Sort by posting rank and segment number, then cut the named episode between the start and end timecodes. Duration is in seconds.
- **Ranked singles** and **Trailers** export the corresponding items in CSV for spreadsheets or JSON for code.
- **Combined promos CSV** joins both lists and identifies each item’s type. **Run report** is the readable rationale and review record.
- **Editor decisions CSV** contains your current-session approvals, rejections and notes.
- **Download everything** bundles these files plus the story bible and evaluation report when available. Until an editor pass exists, the cutting export follows judge order with singles first.''')

    with st.expander('7. Quality and evaluation', expanded=expanded):
        st.markdown('''**Gold labels** are human-selected time spans used for offline comparison. Only episodes with available candidates enter the evaluation. **IoU** divides the shared duration by the union of both spans. The primary IoU rule requires at least 0.5. **Hook match** requires a start within three seconds and IoU of at least 0.3. **Hit@1** asks whether the top clip hits a gold pick in each episode. **Recall@3** asks how many gold picks appear among the top three.

**Chance** is the match rate for one uniformly sampled valid window. **Coverage** is the share of each episode covered by the union of the top three spans. **Confidence intervals** resample episodes to show uncertainty. Random results average 100 seeds. The rules-only baseline uses timing and source rules, and generator self-score uses proposal order without the independent judge. Start error measures time difference among matched spans. Blind ratings can be compared with saved ranks when available.''')

    with st.expander('8. Limits and privacy', expanded=expanded):
        st.markdown('''Screenplay timing can differ from final video. Check the real footage before cutting. Setup-heavy hooks, ambiguous character context and poorly connected trailer segments can slip through. Model proposals vary across requests, and safety labels still need human review. Free-tier quotas can stop a run. Uploads and their artifacts stay in a temporary folder for the session. Live runs send the relevant screenplay content to the selected providers. **Clear session** removes temporary files, pasted keys and session decisions.''')

    with st.expander('9. FAQ', expanded=expanded):
        st.markdown('''**Why four episodes in the demo?** They give a short fictional pack with singles, a cross-episode trailer and an example evaluation.

**Where are the real 50-episode results?** Select **Load results bundle (.zip)** and upload `raw_outputs.zip` from the submission link. The ZIP is private and is provided separately from the public app.

**Why did a clip’s start change?** The validator may move it to include a complete line or put the hook inside the opening window. Open the validator notes for the exact change.

**Why is a clip marked caution?** Read the judge’s reason and citation, then check the footage before approving.

**Do I need a key?** Demo, mock and saved-run browsing work without keys. Live upload generation needs keys for its configured providers.

**Why do I see the same results after switching model?** Results resume only when the files, selected stages and models, and prompt versions match a saved run. A different model opens a separate run. Use **Start fresh run** for a clean copy with the same settings.

**Which providers work on the hosted app?** Gemini, Groq and OpenAI work with valid keys. The provider list marks missing keys. Ollama is local-only. Groq candidates run one episode at a time.

**Are my API keys stored?** Pasted keys stay in this browser session’s app state. They override hosted keys and are never written to run files or logs. **Clear session** removes them. Hosted limits lift only when every active provider that needs a key uses your own key.

**How do I get timestamps into my editor?** Copy an individual timecode range from its card or download the editor-ready CSV for the whole posting queue.''')
