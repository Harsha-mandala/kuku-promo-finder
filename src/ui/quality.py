"""Present saved offline evaluation for the active bundle only."""

import json

import pandas as pd
import streamlit as st

from ..eval import comparison_rows, primary_comparison_rows, percent
from ..eval_cost import cost_report


def show_quality(bundle, config=None):
    st.subheader("Quality & evaluation")
    if bundle["tag"] == "demo":
        st.info("Demo data — fictional gold picks and MockProvider scores. These numbers do not measure a real show or model.")
    path = bundle["output"] / f"eval_{bundle['tag']}/metrics.json"
    if not path.exists():
        st.info("No saved evaluation for this run. Run python -m src.eval run --tag <tag> with this run’s parsed/output directories. Evaluation makes no API calls.")
        if (bundle['output']/'llm_usage.csv').is_file():
            models = (config or {}).get('models', {})
            candidate_provider = models.get('candidates', {}).get('provider')
            judge_provider = models.get('judge', {}).get('provider')
            cost = cost_report(bundle['output'], bundle['tag'], {promo.episode for promo in bundle['validated']},
                               candidate_provider, judge_provider)
            st.markdown('**Cost**')
            st.caption(cost['scope'])
            if cost['stages']:
                st.dataframe(pd.DataFrame(cost['stages']), hide_index=True, width='stretch')
        return
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        st.error(f"Could not read saved evaluation: {error}")
        return
    st.markdown("**System comparison · hook match and IoU**")
    st.caption("The lenient shorter-clip rule let broad windows match without finding the same hook. IoU and hook alignment are now primary; random is averaged over 100 seeds.")
    st.caption("Chance H/R means a single uniformly sampled valid window matching any gold pick / the mean share of gold picks. It is not the chance of three recommendations.")
    st.dataframe(pd.DataFrame(primary_comparison_rows(metrics) if 'primary' in metrics else comparison_rows(metrics)), hide_index=True, width="stretch",
                 alt="Offline system comparison with episode-bootstrap confidence intervals")
    gold = metrics["gold"]
    st.caption(f"Gold coverage: n={len(gold['evaluated_episodes'])} episodes, {gold['evaluated_rows']} picks. "
               f"Pending gold episodes: {gold['pending_episodes']}. 95% intervals resample episodes; small n means substantial uncertainty.")
    with st.expander("Lenient secondary comparison · 50% shorter-clip overlap"):
        st.dataframe(pd.DataFrame(comparison_rows(metrics)), hide_index=True, width="stretch")
    health = metrics["health"]
    st.markdown("**Pipeline health**")
    st.dataframe(pd.DataFrame([{"Check": key, "Compliant": item["count"], "n": item["n"],
                               "Share": percent(item["share"])} for key,item in health["before_validator"].items()]),
                 hide_index=True, width="stretch")
    counts = [{"Kind":"Outcome", "Check":name, "Count":count}
              for name,count in health["validator"]["outcomes"].items()]
    counts.extend({"Kind":"Flag", "Check":name, "Count":count}
                  for name,count in health["validator"]["flag_counts"].items())
    st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
    st.caption(f"Hook reassigned: {percent(health['validator']['hook_reassigned_share'])}; "
               f"identical-score-vector ties: {percent(health['judge']['identical_score_vector_tie_share'])}.")
    st.markdown('**Judge score histograms**')
    st.dataframe(pd.DataFrame([{"Dimension": k.replace('_', ' ').title(),
                               "Score → count": ' · '.join(f'{score} → {count}' for score, count in v.items())}
                              for k,v in health["judge"]["distributions"].items()]),
                 hide_index=True, width="stretch")
    safety = ", ".join(f"{name}: {count}" for name,count in health["judge"]["ad_safety"].items())
    st.caption(f"Safety — {safety}; unverified: {health['judge']['unverified']}.")
    st.markdown("**Judge versus human**")
    human = metrics["human"]
    if not human["n_rated"]:
        st.info("No human ratings yet. Fill my_rating (1–5) in the blind CSV and rerun the offline evaluation.")
    else:
        st.dataframe(pd.DataFrame([{"Compared with":field, "n":human[field]["n"],
                     "Spearman":human[field]["spearman"],
                     "Pairwise agreement":percent(human[field]["pairwise_agreement"]["share"])}
                    for field in ("final_score", "post_priority")]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(human["biggest_disagreements"]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([{'Order':key,**human[key]} for key in ('judge_order','editor_order','final_order') if key in human]),hide_index=True,width='stretch')
    st.markdown("**Cost**")
    st.caption(health["cost"]["scope"])
    if health["cost"]["stages"]:
        st.dataframe(pd.DataFrame(health["cost"]["stages"]), hide_index=True, width="stretch")
    else:
        st.caption("No recorded live usage for this bundle.")
    with st.expander("Per-episode allocated cost"):
        st.caption(health["cost"].get("allocation_note", "No usage log available."))
        st.dataframe(pd.DataFrame(health["cost"]["per_episode"]), hide_index=True, width="stretch")
    with st.expander("Failure gallery"):
        if metrics["failures"]:
            rows = [{k:json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()}
                    for row in metrics["failures"]]
            frame = pd.DataFrame(rows)
            if "episode" in frame:
                frame["episode"] = frame["episode"].astype("string")
            st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.caption("No failure-gallery entries in this evaluation scope.")
    with st.expander("Gemini versus Groq · shared episodes"):
        rows = [{k:json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()}
                for row in metrics["model_comparison"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No saved provider comparison for this bundle.")
    report = path.with_name("eval_report.md")
    if report.exists():
        st.download_button("Download evaluation report", report.read_bytes(), file_name=f"eval_{bundle['tag']}.md", mime="text/markdown")
