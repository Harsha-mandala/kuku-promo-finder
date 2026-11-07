import csv
import json
from pathlib import Path

import pytest

from src.baselines import build_baseline
from src.eval import export_rating_sheet, failure_gallery, load_gold, load_saved, run_eval, system_inputs
from src.eval_cost import cost_report
from src.eval_metrics import bootstrap_ci, evaluate_system, matches, pairwise_agreement, spearman
from src.models import Episode, Line, Scene, Shot
from src.parser import parse_file
from src.validate import ValidatedPromo


def pick(episode, start, end, **extra):
    return {"episode": episode, "span": (start, end), **extra}


def test_overlap_49_vs_51_percent_and_episode_identity():
    gold = pick(1, 0, 100)
    assert not matches(pick(1, 51, 151), gold, rule="lenient")
    assert matches(pick(1, 49, 149), gold, rule="lenient")
    assert matches(pick(1, 50, 150), gold, rule="lenient")
    assert not matches(pick(2, 49, 149), gold, rule="lenient")
    assert matches(pick(1, 70, 170), gold, .3, rule="lenient")


def test_hit_recall_and_start_error_on_handmade_data():
    gold = [pick(1, 0, 20), pick(1, 40, 60), pick(2, 0, 20)]
    promos = [pick(1, 1, 21), pick(1, 40, 60), pick(2, 40, 60), pick(2, 2, 22)]
    result = evaluate_system(promos, gold, samples=50)
    assert result["n_episodes"] == 2 and result["n_gold"] == 3
    assert result["hit_at_1"] == .5 and result["recall_at_3"] == 1
    assert result["mean_start_error_s"] == 1
    assert result["start_within_3s_share"] == 1


def test_bootstrap_reproducible_and_empty_metrics_are_null():
    episodes = [{"hit_at_1": 1, "matched_gold": 2, "n_gold": 2},
                {"hit_at_1": 0, "matched_gold": 0, "n_gold": 1}]
    first = bootstrap_ci(episodes, seed=7, samples=100)
    assert first == bootstrap_ci(episodes, seed=7, samples=100)
    assert first["hit_at_1"] == [0, 1]
    assert evaluate_system([], [], samples=20)["hit_at_1"] is None
    assert bootstrap_ci([])["recall_at_3"] is None


@pytest.mark.parametrize("system", ["random", "rules_only"])
def test_baselines_use_clean_disjoint_valid_shots_on_synthetic_fixture(system):
    episode = parse_file(Path(__file__).parent/"fixtures/sample_episode.txt")
    promos = build_baseline(episode, system, seed=7)
    assert promos and len(promos) <= 3
    assert [p.model_dump() for p in promos] == [p.model_dump() for p in build_baseline(episode, system, seed=7)]
    for i, promo in enumerate(promos):
        assert 20 <= promo.duration_s <= 60 and promo.start_clean and promo.end_clean
        assert ValidatedPromo.model_validate_json(promo.model_dump_json()) == promo
        for other in promos[:i]:
            assert min(promo.final_span[1], other.final_span[1]) <= max(promo.final_span[0], other.final_span[0])


def make_saved(tmp_path, numbers=(1, 2, 7)):
    config = {"parsed_dir":str(tmp_path/"parsed"), "output_dir":str(tmp_path/"outputs"),
              "models":{"candidates":{"provider":"gemini"}, "judge":{"provider":"groq"}},
              "evaluation":{"gold_csv":str(tmp_path/"eval/gold.csv"), "gold_episodes":[7, 19],
                            "seed":5,"bootstrap_samples":50}}
    parsed, output = Path(config["parsed_dir"]), Path(config["output_dir"])
    parsed.mkdir()
    ranked, scores = [], []
    for number in numbers:
        lines, shots = [], []
        for i in range(12):
            line = Line(id=f"E{number:03d}_L{i+1:03d}", episode=number, scene=1, type="dialogue",
                        speaker="TEST", emotion="Shocked", start=i*5+.2, end=i*5+1, text=f"Synthetic question {i}?")
            lines.append(line)
            shots.append(Shot(id=f"E{number:03d}_S{i+1:03d}", index=i+1, episode=number, scene=1,
                              start=i*5,end=(i+1)*5,camera_text="Synthetic close-up.",lines=[line]))
        ep = Episode(number=number,source_file="synthetic",duration=60,shots=shots,lines=lines,
                     scenes=[Scene(number=1,heading="TEST",start=0,end=60)])
        (parsed/f"episode_{number:03d}.json").write_text(ep.model_dump_json(),encoding="utf-8")
        candidates, promos = [], []
        for i in range(3):
            key=f"E{number:03d}_C{i+1:02d}"
            promo=ValidatedPromo(episode=number,id=key,status="ok",start_shot_id=shots[i*4].id,
                end_shot_id=shots[i*4+3].id,final_span=(i*20,(i+1)*20),duration_s=20,self_score=5-i,
                start_clean=True,end_clean=True,hook_text="Synthetic hook?",cliffhanger_text="Synthetic cliff?",
                why_it_works="SECRET_GENERATOR_NOTE")
            promos.append(promo.model_dump())
            candidates.append({"candidate_id":key,"span_start":i*20,"span_end":(i+1)*20,
                               "self_score":5-i,"hook_offset_s":.2,"start_clean":True,"end_clean":True})
            ranked.append({"promo_id":key,"rank":len(ranked)+1,"episode":number,"final_score":.9-i*.2,
                           "segments":[{"episode":number,"start":i*20,"end":(i+1)*20}]})
            scores.append({"promo_id":key,"hook_score":4,"cliffhanger_score":4,"standalone_score":4,
                           "emotion_score":4,"post_priority":8-i,"ad_safety":"safe","weakest_point":"Synthetic weakness."})
        for folder, data in (("candidates",candidates),("validated",promos)):
            directory=output/f"{folder}_test";directory.mkdir(exist_ok=True,parents=True)
            (directory/f"episode_{number:03d}.json").write_text(json.dumps(data),encoding="utf-8")
    for folder, name, data in (("final","promos_ranked.json",ranked),("judged","scores.json",scores)):
        directory=output/f"{folder}_test";directory.mkdir(exist_ok=True)
        (directory/name).write_text(json.dumps(data),encoding="utf-8")
    gold=Path(config["evaluation"]["gold_csv"]);gold.parent.mkdir()
    gold.write_text("episode,start,end,hook,why\n7,00:00.00,00:20.00,Synthetic hook,Synthetic reason\n19,00:00.00,00:20.00,Pending,Pending\n",encoding="utf-8")
    return config


def test_partial_coverage_is_shared_and_rerunnable_without_api(tmp_path, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm,"get_llm",lambda *a,**k:pytest.fail("No clients in offline evaluation"))
    config=make_saved(tmp_path)
    result=run_eval(config,"test",eval_dir=tmp_path/"eval")
    assert result["gold"]["evaluated_episodes"] == [7]
    assert result["gold"]["pending_episodes"] == [19]
    assert all(v["0.5"]["n_episodes"]==1 for v in result["systems"].values())
    assert result["health"]["n_candidates"] == 9
    assert result["api_calls_made"] == 0
    again=run_eval(config,"test",eval_dir=tmp_path/"eval")
    assert again["systems"] == result["systems"]
    assert run_eval(config,"test",eval_dir=tmp_path/"eval",health_only=True)["gold"]["evaluated_rows"] == 0


def test_gold_bad_rows_warn_without_crashing(tmp_path):
    config=make_saved(tmp_path)
    path=Path(config["evaluation"]["gold_csv"])
    with path.open("a",encoding="utf-8") as stream:
        stream.write("7,bad,00:20.00,a,b\n7,00:20.00,00:01.00,a,b\n7,00:50.00,01:20.00,a,b\n7,00:00.00,00:10.00,a,b\n")
    result=load_gold(path,load_saved(config,"test")["episodes"])
    assert result["rows_found"] == 6 and result["rows_valid"] == 1
    assert len(result["warnings"]) == 5


def test_blind_sheet_excludes_gold_and_is_deterministic_without_leaks(tmp_path):
    config=make_saved(tmp_path)
    first=export_rating_sheet(config,"test",n=5,eval_dir=tmp_path/"eval")
    path=Path(first["path"])
    before=path.read_bytes()
    second=export_rating_sheet(config,"test",n=5,eval_dir=tmp_path/"eval")
    assert path.read_bytes() == before and first["promo_ids"] == second["promo_ids"]
    assert first["promo_ids"] != sorted(first["promo_ids"])
    with path.open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==5 and not {7,19}.intersection(int(r["episode"]) for r in rows)
    assert set(rows[0]) == {"promo_id","episode","span","duration","clip_script","my_rating"}
    assert "SECRET_GENERATOR_NOTE" not in before.decode()
    rows[0]["my_rating"]="4"
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="human ratings"):
        export_rating_sheet(config,"test",n=5,eval_dir=tmp_path/"eval")


def test_capped_promos_follow_judged_for_recall_only():
    promos=[ValidatedPromo(episode=1,id=f"E001_C{i:02d}",status="ok",final_span=((i-1)*20,i*20),self_score=5)
            for i in range(1,6)]
    saved={"candidates":[],"ranked":[{"rank":1,"promo_id":promos[0].id,"segments":[{"episode":1,"start":0,"end":20}]},
                                     {"rank":2,"promo_id":promos[1].id,"segments":[{"episode":1,"start":20,"end":40}]}],
           "promos":promos,"available":[1],"scores":[],"exclusions":{"capped":[{"id":promos[3].id},{"id":promos[4].id}]}}
    full=system_inputs(saved)["full_pipeline"]
    assert full[2]["id"]=="E001_C04" and full[2]["recall_only"]
    metrics=evaluate_system(full,[pick(1,60,80)],samples=10)
    assert metrics["hit_at_1"] == 0 and metrics["recall_at_3"] == 1


def test_spearman_ties_and_pairwise_agreement():
    assert spearman([1,2,3],[3,2,1]) == pytest.approx(-1)
    assert spearman([1,1,3],[2,2,4]) == pytest.approx(1)
    assert spearman([1,1],[2,3]) is None
    assert pairwise_agreement([1,2],[3,3])["share"] == .5


def test_cost_intervals_exclude_other_runs_and_label_batch_allocation(tmp_path):
    events=[{"stage":"candidates","state":"running","time":"2026-11-06T01:00:00+00:00"},
            {"stage":"candidates","state":"complete","time":"2026-11-06T01:02:00+00:00"}]
    (tmp_path/"pipeline_test_events.jsonl").write_text("\n".join(json.dumps(e) for e in events),encoding="utf-8")
    with (tmp_path/"llm_usage.csv").open("w",encoding="utf-8",newline="") as stream:
        fields=["time","tag","provider","cached","in_tokens","out_tokens","latency_s"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for time in ("01:01:00","03:01:00"):
            writer.writerow(dict(time=f"2026-11-06T{time}+00:00",tag="candidates_batch_E001_E002",provider="gemini",cached="False",in_tokens=100,out_tokens=20,latency_s=5))
    result=cost_report(tmp_path,"test",{1,2},"gemini","groq")
    assert result["stages"][0]["api_calls"]==1
    assert result["per_episode"][0]["input_tokens_known_allocated"]==50
    assert result["stage_wall_s"]["candidates"]==120


def test_failure_annotations_survive_rerun_and_human_agreement(tmp_path):
    config=make_saved(tmp_path)
    exported=export_rating_sheet(config,"test",n=6,eval_dir=tmp_path/"eval")
    path=Path(exported["path"])
    with path.open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    for i,row in enumerate(rows):
        row["my_rating"]=str(i%5+1)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=run_eval(config,"test",eval_dir=tmp_path/"eval")
    assert result["human"]["n_rated"]==6
    assert result["human_by_sheet"]["whole_range"]["n_rated"] == 6
    assert result["human_by_sheet"]["top_of_list"]["n_rated"] == 0
    assert result["human_by_sheet"]["both"]["n_rated"] == 6
    assert len(result["human"]["biggest_disagreements"])==3
    failure=tmp_path/"eval/failures_test.csv"
    with failure.open(encoding="utf-8",newline="") as stream:
        failures=list(csv.DictReader(stream))
    failures[0]["my_reason"]="My own explanation."
    with failure.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(failures[0]));writer.writeheader();writer.writerows(failures)
    again=run_eval(config,"test",eval_dir=tmp_path/"eval")
    assert again["failures"][0]["my_reason"]=="My own explanation."


def test_new_saved_gold_episode_is_picked_up_on_next_run(tmp_path):
    import shutil
    first=tmp_path/'first';first.mkdir()
    config=make_saved(first)
    before=run_eval(config,'test',eval_dir=first/'eval')
    other=tmp_path/'other';other.mkdir()
    more=make_saved(other,numbers=(19,))
    for source, destination in ((Path(more['parsed_dir']),Path(config['parsed_dir'])),
                                (Path(more['output_dir'])/'candidates_test',Path(config['output_dir'])/'candidates_test'),
                                (Path(more['output_dir'])/'validated_test',Path(config['output_dir'])/'validated_test')):
        for path in source.glob('episode_*.json'):
            shutil.copyfile(path,destination/path.name)
    after=run_eval(config,'test',eval_dir=first/'eval')
    assert before['gold']['evaluated_episodes']==[7]
    assert after['gold']['evaluated_episodes']==[7,19]
    assert all(v['0.5']['n_episodes']==2 for v in after['systems'].values())


def test_baseline_choices_are_independent_of_gold(tmp_path):
    config=make_saved(tmp_path)
    run_eval(config,'test',eval_dir=tmp_path/'eval')
    baseline=Path(config['output_dir'])/'eval_test/baselines/rules_only/episode_007.json'
    before=baseline.read_bytes()
    Path(config['evaluation']['gold_csv']).write_text('episode,start,end,hook,why\n7,00:30.00,00:55.00,Changed,Changed\n',encoding='utf-8')
    run_eval(config,'test',eval_dir=tmp_path/'eval')
    assert baseline.read_bytes()==before
