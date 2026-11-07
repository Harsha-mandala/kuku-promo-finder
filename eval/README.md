# Offline evaluation

Run with `.venv\Scripts\python.exe` and `PYTHONIOENCODING=utf-8`:

```powershell
.venv\Scripts\python.exe -m src.eval run --tag gemini_full
.venv\Scripts\python.exe -m src.eval run --tag groq --health-only
.venv\Scripts\python.exe -m src.eval export-rating-sheet --tag gemini_full --n 20
.venv\Scripts\python.exe -m src.eval export-rating-sheet --tag gemini_full --top --n 10
```

Repeat the first command after more saved episodes exist. Missing gold episodes
are listed, not scored as misses. Every system uses the same available gold
episodes, including successfully saved empty candidate lists (zero hits).
No provider client is created. `gold.csv` and human rating/failure sheets are
private; never put them in the public/demo package.

Gold input is `episode,start,end,hook,why`. Bad rows are reported and excluded.
The ten designated holdout episodes live in `evaluation.gold_episodes` in
config.yaml; all are excluded from blind-sheet sampling, even before candidates
or valid gold rows exist. Human ratings use `my_rating`, integer 1–5. Existing
ratings are protected from overwrite unless `--force` is explicitly requested.
The top sheet samples the saved editor pool and excludes clips already on the
whole-range sheet and clips touching any gold episode. After rating it, rerun
evaluation to see agreement for each sheet and for the combined ratings.
Failure `my_reason` annotations survive evaluation reruns.

## Fixed baselines

These rules were defined before computing gold results. They do not learn from
gold.csv. All keywords and weights are in `evaluation.baseline_rules`:

- Emotions: shocked, outraged, furious, cruel, gloating, screaming, mocking,
  venomous, desperate. Match complete words in dialogue emotion tags.
- Action/sound word roots: slap, strike, punch, gasp, scream, grab, drag, kneel.
  Word-root matching accepts suffixes such as “slaps” and “screaming”.
- Hook: +3 when the first three seconds contain qualifying dialogue emotion or
  action/sound words.
- Cliffhanger: +2 if the last line ends with ?/!, has a listed emotion, or the
  window ends on the final non-dead shot.
- Conflict: +0.25 per distinct speaker, capped at three speakers.
- Exclamation density: +0.5 times exclamation marks / word count.

Rules-only enumerates 25–45-second clean whole-shot windows, greedily selecting
the top three disjoint spans; ties prefer earlier start, then earlier end.
Random shuffles all 20–60-second clean windows with seed + episode number and
greedily selects up to three disjoint spans. Fewer windows are allowed when the
episode cannot supply three. This is a reproducible sanity floor, not a uniform
sample over sets of three windows. Neither baseline is subsequently auto-fixed.
They are saved as ValidatedPromo records under outputs/eval_TAG/baselines/.

## Metrics and limits

Primary match = same episode and intersection / union (IoU) >= 0.50.
Hook match = same episode, start distance <=3 s, and IoU >=0.30.
The old overlap / shorter duration >=0.50 (also 0.30) is secondary and labelled
lenient: broad windows can cover a gold moment without finding its boundaries.
One tool proposal may match multiple gold picks. Hit@1 averages episode hits;
Recall@3 pools matched gold rows. For each matched gold pick, use the closest
start among matching top-three proposals for mean absolute start error and ±3 s.
Random metrics average 100 distinct seeds drawn from the master seed, reporting
mean ± population SD. Top-three coverage is the union of selected spans divided
by episode duration, averaged over available gold episodes (and seeds for random).
Chance levels enumerate all clean 20–60 s windows exactly: probability of one
uniformly sampled window matching any gold pick (Hit), or mean per-pick match
probability (Recall), averaged over episodes. These are NOT top-three chance
probabilities; the repeated random baseline measures that separately.
Bootstrap resamples episodes with replacement 2000 times using the fixed seed;
recall retains each resampled episode’s number of gold picks. Random intervals
resample per-episode averages across seeds, separate from seed SD. Small n is fragile.

Generator self-score uses original proposals, before repairs. Full pipeline uses
final validated spans and saved final editor order when available, otherwise
judge order; trailers are evaluated separately from single-window gold picks.
Promos not judged due to the cap
follow judged promos by self_score for Recall@3; they cannot supply an unjudged
Hit@1. Rejected and unsafe promos are not full-pipeline recommendations.

Agreement uses tied midranks for Spearman. Pairwise agreement excludes human
ties and awards half credit for model ties. Undefined correlations are null.
The failure gallery treats the first 15 global ranks as “top-ranked”. The nearest
promo for a missed gold pick maximizes overlap, then minimizes start distance.
Human agreement additionally correlates ratings with negative judge/editor/final
ranks (higher means preferred); without human ratings the results remain null.

Usage CSV does not contain run tags. Pipeline event intervals provide temporal
tag attribution when available, excluding pings. Legacy output without those
events reports related provider/episode history, explicitly not exact tag cost.
Batch costs are divided equally across requested episodes and labelled estimated;
missing provider usage is unknown, never assumed to be zero. CSV latency is
provider-call duration; pipeline stage wall time also includes waiting.

Demo gold is entirely fictional. Demo scores are authored MockProvider responses;
the demo evaluation demonstrates calculations and is not evidence of model quality.
