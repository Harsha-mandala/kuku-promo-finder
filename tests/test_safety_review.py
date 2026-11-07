from src.safety_review import review_queue


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
