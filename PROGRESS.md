# Public progress log

Private run reports, screenplay examples and previous detailed logs are preserved
under ignored outputs and are not part of the public/demo package.

- Step 1: screenplay parsing, shot assignment, effective spans, warnings, CLI and synthetic tests.
- Step 2: swappable providers, caching, retries, immutable raw logs and quota handling.
- Step 3: source-ID candidates, optional empty hints, story bible, compact batches and dry-run planning.
- Step 4: whole-shot validation/repair, overlap dedupe, shared call budgets and resume.
- Step 5: blind interleaved judging, structured clip views, deterministic settings,
  judge-authored rationales, configurable weights and consecutive soft-diversity ranking.
- Step 6: offline gold evaluation, fixed random/rules-only baselines, episode-bootstrap intervals, blind human sheets, pipeline health/cost audit and failure gallery.
- Step 7: Embers Streamlit editor, fictional four-episode MockProvider demo,
  session-isolated uploads/keys, hosted limits, downloads and editor decisions.
  Added post-priority scoring and grounded safety citation verification; legacy
  rankings explicitly mark the fallback formula. No live API calls in this step.

The CLI remains a thin wrapper over the same pipeline called by the UI. Demo mode
is read-only and independent of confidential local data. Source/demo publication
uses an allowlisted manifest with a private-text audit. Deployment remains manual.

Step 7 verification: `201 passed in 8.57s`, including MockProvider integration,
CLI dry-run compatibility, Streamlit demo/editor/upload-mode smoke tests, password
reset, temporary-folder deletion, hosted caps and safety-citation fallback.
Four demo-only screenshots were captured and inspected. Live API calls: zero.
The fictional demo has eleven promos, including one whole-shot validator fix.
Evaluation remains pending; safety citation checks do not prove semantic accuracy.

Latest revision: independent provider budgets and resumable partial-stage continuation; Embers palette with AA body/button contrast, working header copy control, and a realistic 11-promo synthetic demo. Mock regression suite: 203 tests pass. Private live run statistics remain under ignored outputs.

Reliability revision: configurable 180-second Gemini timeout and candidate thinking
budget (1024), 30-second exponential busy/timeout backoff, a three-attempt ceiling,
and failed five-episode batch splitting into three plus two. Every Gemini attempt
counts against its provider budget, including transport failures. Candidate rules
exclude setup and weak opening beats. Validation flags sound/camera-only hooks
and trims trailing footage after the cited cliffhanger when at least 20 seconds
remain. The public MockProvider demo now has unique hooks, cliffhangers and
judge rationales for every promo; the candidate grid has a narrower default view
with optional columns. Private resume statistics remain under ignored outputs.

Reliability revision verification: `213 passed in 8.42s`; public/private text audit passed.

Step 6 complete: overlap metrics at 50% and 30%, original-generator versus
validated-pipeline ranking, cap-aware Recall@3, incremental gold coverage,
reproducible bootstrap, human correlation/agreement, comparison across saved
providers, and annotation-preserving failure exports. All evaluation is offline.
The Quality tab shows only the active bundle’s metrics and marks fictional demo
results clearly. Gold/rating files remain private and excluded from packaging.
Tests: `228 passed in 8.66s`. No live API calls. Private results and partial-coverage
limitations are documented under outputs/eval_<tag>/; no general quality claim
is made from a small gold sample or the authored MockProvider demo.

Step 8 implemented offline: IoU/hook-first evaluation with analytical chance,
100-seed random means/SD, coverage and episode-bootstrap intervals; separate
lenient comparison retained. Added source-ID multi-segment trailer proposals,
per-segment whole-shot validation, blind coherence judging, separate trailer
ranking, combined exports and a blinded comparative editor pass with configured
rank blending. New stages share provider budgets and save resumable outputs.
Selective candidate refresh uses --force-episodes while preserving the bible.
The fictional MockProvider demo exercises a cross-episode trailer and editor
rank disagreement. UI adds segment copy controls and dashed trailer tracks.
No live APIs were called. Private evaluation/dry-run/status details stay under
ignored outputs; no real-show trailer/editor result is fabricated.

Step 8 verification: `247 passed in 10.36s`; targeted Hide caution trailer filtering passed. The trailer segment copy control and dashed timeline were visually checked. No live API calls.

Final Step 8 verification: `249 passed in 9.99s`, including stale-source protection, rank-only editor cache reuse, new-provider credential resolution and Arrow-safe tables. Public/private text audit: pass. No live APIs called.

Step 9 completed the gemini_full live run and offline UI upgrade. The validator now
keeps reassigned hooks within the opening window after boundary repairs. The regression
fixture contains E018 timings and IDs only. The judge prompt stayed unchanged.
The completed run covers 50 episodes. Evaluation covers all 10 gold episodes, and
the blind sheet contains 20 non-gold promos. Private run details are under outputs/.

The UI now offers a loopback-only saved-run view, mock uploads, segment-level
editor exports, a ZIP, on-bar trailer overlays, Quality summaries and an expanded
user guide. Public deployments hide saved-run mode. Demo outputs use fictional
episodes and MockProvider scores.
Final verification: 255 passed in 37.36s. Public audit passed. Five UI screenshots are saved under ignored outputs/ui_screens/.

Step 10 validated 20 blind ratings and added clip-bootstrap agreement intervals
to evaluation. The editor comparison has one rated clip in common with the
original editor order. The completed trailer refresh ranked three trailers
under the new ten-second same-episode gap rule. The single permitted Gemini
editor attempt omitted two IDs, so its response was rejected and the current
posting shortlist uses judge order. The original editor artifacts are preserved.
An offline keyword review queue and safe-only posting list now appear in the
report and UI. Citation-ready run facts are saved privately under outputs/.
Verification: 257 tests passed with a socket-free Streamlit AppTest loop in the
test harness. The private-text audit passed.

Step 11 added editor fallback handling and separate agreement for whole-range and
top-of-list blind sheets. The live editor restored the blended order for the main
run. A separate fictional Groq smoke test completed parsing, context, generation,
validation, judging, ranking and trailers. Per-minute rate limits exhausted its
call budget during the editor stage. The bundled mock demo remains available
offline. The private submission archive and the public source package are
separate. Interview commands and demo screenshots are included in the public
package.
