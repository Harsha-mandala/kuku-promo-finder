# Evaluation — demo

## System comparison (primary: hook match and IoU)

| System | Hook-Hit@1 | Hook-Recall@3 | IoU-Hit@1 | IoU-Recall@3 | Start error (s) | Chance hook H/R | Chance IoU H/R | n episodes | n gold | Coverage top3 | Hook CI95 Hit / Recall | IoU CI95 Hit / Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 13.0% ± 18.5% | 29.5% ± 24.6% | 23.2% ± 20.7% | 57.2% ± 25.3% | 6.78 ± 5.66 | 11.0% / 11.0% | 20.6% / 20.6% | 4 | 4 | 82.0% ± 5.7% | [8.0%, 17.0%] / [20.0%, 39.0%] | [19.0%, 29.0%] / [41.5%, 73.0%] |
| rules_only | 25.0% | 50.0% | 50.0% | 75.0% | 4.00 | 11.0% / 11.0% | 20.6% / 20.6% | 4 | 4 | 95.8% | [0.0%, 75.0%] / [0.0%, 100.0%] | [0.0%, 100.0%] / [25.0%, 100.0%] |
| generator_self_score | 50.0% | 75.0% | 50.0% | 100.0% | 2.50 | 11.0% / 11.0% | 20.6% / 20.6% | 4 | 4 | 93.8% | [0.0%, 100.0%] / [25.0%, 100.0%] | [0.0%, 100.0%] / [100.0%, 100.0%] |
| full_pipeline | 50.0% | 100.0% | 50.0% | 100.0% | 0.00 | 11.0% / 11.0% | 20.6% / 20.6% | 4 | 4 | 91.7% | [0.0%, 100.0%] / [100.0%, 100.0%] | [0.0%, 100.0%] / [100.0%, 100.0%] |

The shorter-clip overlap rule was replaced because long windows can cover a gold moment without agreeing on its boundaries or hook.

Chance is analytical for ONE uniformly sampled clean 20–60 s window, averaged over gold episodes; it is not the chance of three recommendations. Random uses 100 seeds: mean ± population SD. Its bootstrap resamples episode means after averaging seeds.

| Rule | Hit_chance | Recall_chance |
| --- | --- | --- |
| hook | 11.0% | 11.0% |
| iou | 20.6% | 20.6% |
| lenient | 42.6% | 42.6% |

## Lenient secondary comparison (50% shorter-clip overlap)

| System | n episodes | n gold | Hit@1 | Hit CI95 | Recall@3 | Recall CI95 | Start error (s) | Start ±3s | Matched pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 4 | 4 | 50.0% ± 23.7% | [43.0%, 62.5%] | 91.2% ± 13.9% | [81.8%, 98.0%] | 8.46 | 32.6% | 3.65 |
| rules_only | 4 | 4 | 50.0% | [0.0%, 100.0%] | 75.0% | [25.0%, 100.0%] | 4.00 | 66.7% | 3 |
| generator_self_score | 4 | 4 | 50.0% | [0.0%, 100.0%] | 100.0% | [100.0%, 100.0%] | 2.50 | 75.0% | 4 |
| full_pipeline | 4 | 4 | 50.0% | [0.0%, 100.0%] | 100.0% | [100.0%, 100.0%] | 0.00 | 100.0% | 4 |

Available gold: [1, 2, 3, 4] (n=4, 4 gold promos). Pending gold episodes: [].
These are partial-coverage results, not performance over all gold episodes or all 50 episodes.

95% percentile bootstrap intervals resample episodes, preserving all gold picks within each episode. Small n gives unstable uncertainty estimates. Missing gold episodes are not scored as misses. Recall@3 is gold-row weighted; Hit@1 is episode weighted.

An all-success sample can yield a degenerate [100%, 100%] bootstrap interval; this does not establish certainty on unseen episodes.

## Looser comparison (30% shorter-clip overlap)

| System | n episodes | n gold | Hit@1 | Hit CI95 | Recall@3 | Recall CI95 | Start error (s) | Start ±3s | Matched pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 4 | 4 | 50.5% ± 23.7% | [43.5%, 63.0%] | 92.5% ± 12.5% | [82.0%, 99.0%] | 8.68 | 31.7% | 3.7 |
| rules_only | 4 | 4 | 50.0% | [0.0%, 100.0%] | 75.0% | [25.0%, 100.0%] | 4.00 | 66.7% | 3 |
| generator_self_score | 4 | 4 | 50.0% | [0.0%, 100.0%] | 100.0% | [100.0%, 100.0%] | 2.50 | 75.0% | 4 |
| full_pipeline | 4 | 4 | 50.0% | [0.0%, 100.0%] | 100.0% | [100.0%, 100.0%] | 0.00 | 100.0% | 4 |

Each gold pick may match any top-three proposal; matches are not one-to-one. Start error uses the closest start among matching top-three proposals for each gold pick. Generator self-score evaluates original proposals; full pipeline uses final validated spans. Capped promos follow judged promos by self_score for Recall@3 only. Unsafe/rejected promos are excluded from the full pipeline.

## Pipeline health

4 episodes; 11 original saved candidates.

| Before validator | n | Compliant | Share |
| --- | --- | --- | --- |
| duration_20_60 | 11 | 11 | 100.0% |
| hook_within_3s | 11 | 10 | 90.9% |
| clean_start | 11 | 11 | 100.0% |
| clean_end | 11 | 11 | 100.0% |
| both_clean | 11 | 11 | 100.0% |
| all | 11 | 10 | 90.9% |

Validator outcomes: {'ok': 10, 'fixed': 1, 'flagged': 0, 'rejected': 0}. Flags: {}. Hook reassignment share: 0.0%.

| Fix / audit type | Count |
| --- | --- |
| start moved to hook shot | 1 |

## Judge distributions

| Dimension | Histogram |
| --- | --- |
| hook_score | {3: 2, 4: 5, 5: 4} |
| cliffhanger_score | {4: 7, 5: 4} |
| standalone_score | {3: 2, 4: 5, 5: 4} |
| emotion_score | {3: 2, 4: 5, 5: 4} |
| post_priority | {7: 6, 8: 4, 9: 1} |

Identical-score-vector tie share: 0.0%; denominator is judged promos in duplicate vectors / all judged promos. Legacy missing post_priority: 0. Safety: {'caution': 2, 'safe': 9}; unverified: 1.

## Judge versus human

Valid human ratings: 0. No agreement is claimed before ratings are supplied.

| Compared with | n | spearman | pairwise_agreement |
| --- | --- | --- | --- |
| final_score | 0 | None | {'share': None, 'comparable_pairs': 0, 'model_ties': 0} |
| post_priority | 0 | None | {'share': None, 'comparable_pairs': 0, 'model_ties': 0} |

| Preferred order | n | spearman |
| --- | --- | --- |
| judge_order | 0 | None |
| editor_order | 0 | None |
| final_order | 0 | None |

Order agreement correlates human ratings with negative rank, so a positive coefficient means agreement about which clip to prefer.

Pairwise agreement excludes tied human ratings; model ties receive half credit. Spearman uses average ranks for ties and is undefined for constant or fewer than two observations.

No rows available.

## Cost

No usage log available

No rows available.



### Per-episode allocated cost

No rows available.

Logged stage elapsed seconds: {}

## Gemini versus Groq (shared episodes)

No rows available.

No matching-gold claim is made when n_gold is zero. Legacy prompts and missing priority scores make this an observational comparison, not a controlled model experiment.

## Findings and failures

Full-pipeline IoU-Hit@1 is 50.0%, versus 50.0% for rules-only; IoU-Recall@3 is 100.0% versus 75.0%. Hook-Hit@1 is 50.0%; Hook-Recall@3 is 100.0%. Only 4 full-pipeline gold picks match; 100.0% of their starts are within three seconds. These are descriptive results on available episodes, not proof of superiority or a basis for tuning.

2 failure-gallery entries. Add your analysis to my_reason in the CSV; reruns preserve existing annotations.

| failure_id | kind | episode | promo_id | tool_span | detail | my_reason |
| --- | --- | --- | --- | --- | --- | --- |
| safety_E001_C01 | caution | 1 | E001_C01 | (0.0, 30.0) | Scripted slap at E001_L002. |  |
| safety_E002_C02 | caution | 2 | E002_C02 | (48.0, 76.0) | A spoken threat of fire. |  |

## Rules-only selected windows

| Episode | Span | First line | Last line |
| --- | --- | --- | --- |
| 1 | 00:00.00–00:30.00 | Nera slaps Ivo across the cheek, stopping the public unveiling. | Then whose name did you put on it? |
| 1 | 00:30.00–01:02.00 | Your father signed the transfer himself. | Why is there a photograph of me inside? |
| 1 | 01:02.00–01:40.00 | You sold my identity along with the machine. | The tower bell begins a single unfinished note. |
| 2 | 00:36.00–01:16.00 | Then why does the recording say your name? | What if he never wanted you to find him? |
| 2 | 01:16.00–01:50.00 | You imprisoned him inside his own invention. | The tower bell begins a single unfinished note. |
| 2 | 00:00.00–00:36.00 | That key opens the tower—or your grave. | A voice can be copied. A debt cannot. |
| 3 | 00:16.00–00:56.00 | Who signed it after my father disappeared? | Why is my father bidding against me? |
| 3 | 00:56.00–01:36.00 | The highest bidder ordered my father killed. | The tower bell begins a single unfinished note. |
| 4 | 00:00.00–00:40.00 | That voice on the recording is yours. | Then which of you ordered it to ring? |
| 4 | 00:40.00–01:20.00 | I hid the original beneath the bell. | Who just locked the tower door? |
| 4 | 01:20.00–02:00.00 | You erased my mother from the recording. | The tower bell begins a single unfinished note. |

Baselines are fixed screenplay-only heuristics documented in eval/README.md, not fitted to gold. Random is a seeded sanity floor; either baseline may return fewer than three disjoint valid windows.

## Warnings

None.
