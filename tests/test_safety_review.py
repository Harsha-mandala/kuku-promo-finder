from src.safety_review import review_queue


def test_trailer_safety_inherits_strictest_positive_overlap_and_is_repeatable():
    from src.safety_review import inherit_trailer_safety
    trailer = {"ad_safety": "safe", "segments": [{"episode": 1, "start": 0, "end": 10}]}
    singles = [
        {"promo_id": "A", "ad_safety": "caution", "episode": 1, "final_span": [5, 9]},
        {"id": "B", "ad_safety": "unsafe", "episode": 1, "final_span": [9, 12]},
        {"id": "C", "ad_safety": "unsafe", "episode": 1, "final_span": [10, 15]},
    ]
    effective = inherit_trailer_safety(trailer, singles)
    assert effective["judge_ad_safety"] == "safe"
    assert effective["ad_safety"] == "unsafe"
    assert effective["safety_inherited_from"] == ["B"]
    assert inherit_trailer_safety(effective, singles) == effective


def test_shortlist_keeps_higher_ranked_cut_and_fills_the_next_slot():
    from src.safety_review import shortlist
    trailer = {"promo_id": "T", "type": "trailer", "segments": [
        {"episode": 1, "start": 0, "end": 5}, {"episode": 1, "start": 15, "end": 20}]}
    single = {"promo_id": "S", "episode": 1, "final_span": [0, 20]}
    gap = {"promo_id": "G", "episode": 1, "final_span": [3, 18]}
    inside = {"promo_id": "I", "episode": 1, "final_span": [1, 4]}
    result = shortlist([trailer, single, gap, inside], limit=2)
    assert [row["promo_id"] for row in result] == ["T", "G"]
    assert [row["promo_id"] for row in result[0]["alternate_cuts"]] == ["S", "I"]


def test_queue_keeps_safety_label_and_uses_whole_terms():
    rows = [
        {"promo_id": "review", "ad_safety": "caution", "ad_safety_reason": "Implied intimacy."},
        {"promo_id": "unsafe", "ad_safety": "unsafe", "ad_safety_reason": "A child is threatened."},
        {"promo_id": "safe", "ad_safety": "safe", "ad_safety_reason": "Intimate."},
        {"promo_id": "unrelated", "ad_safety": "caution", "ad_safety_reason": "Minority viewpoint."},
    ]
    queue = review_queue(rows, {}, ["intimacy", "intimate", "minor"], ["threatened"])
    assert [(row["promo_id"], row["ad_safety"]) for row in queue] == [
        ("review", "caution"), ("unsafe", "unsafe")
    ]
    assert all(row["safety_review"] == "needs_human_review" for row in queue)
