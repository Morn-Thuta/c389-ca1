"""Tests for src/counting.py, using hand-made boxes (no model, no images)."""

from src.counting import COUNT_KEYS, count_crossers


def det(name, box=(0, 0, 10, 10), conf=0.9):
    return {"class_name": name, "confidence": conf, "box": list(box)}


def test_empty_scene_has_every_key_at_zero():
    counts, kept = count_crossers([])
    assert list(counts) == COUNT_KEYS
    assert all(v == 0 for v in counts.values())
    assert kept == []


def test_counts_each_class_and_totals_per_direction():
    names = ["person_left", "person_left", "person_right", "wheeled_left", "wheeled_right", "wheeled_right"]
    counts, kept = count_crossers([det(n, (i * 20, 0, i * 20 + 10, 10)) for i, n in enumerate(names)])
    assert counts["person_left"] == 2
    assert counts["person_right"] == 1
    assert counts["wheeled_left"] == 1
    assert counts["wheeled_right"] == 2
    assert counts["total_left"] == 3
    assert counts["total_right"] == 3
    assert [d["id"] for d in kept] == list(range(6))
    assert kept[3]["direction"] == "left"


def test_threshold_drops_low_confidence():
    counts, kept = count_crossers([det("person_right", conf=0.9), det("wheeled_left", conf=0.2)], threshold=0.4)
    assert counts["person_right"] == 1
    assert counts["wheeled_left"] == 0
    assert len(kept) == 1


def test_unknown_classes_are_ignored():
    counts, kept = count_crossers([det("car"), det("bicycle")])
    assert kept == []
    assert sum(counts.values()) == 0


def test_near_identical_boxes_count_once():
    # One cyclist returned twice (IoU 0.96), plus a separate rider beside them.
    counts, kept = count_crossers([
        det("wheeled_right", (1371, 752, 3282, 2480), 0.80),
        det("wheeled_right", (1384, 728, 3274, 2445), 0.60),
        det("wheeled_left", (3400, 700, 3900, 2400), 0.70),
    ])
    assert counts["wheeled_right"] == 1 and counts["wheeled_left"] == 1
    assert kept[0]["confidence"] == 0.80


def test_same_box_both_directions_keeps_the_confident_one():
    counts, _ = count_crossers([det("person_left", (0, 0, 50, 150), 0.5), det("person_right", (2, 1, 51, 150), 0.7)])
    assert counts["person_right"] == 1 and counts["person_left"] == 0


def test_overlapping_people_in_a_crowd_are_both_kept():
    counts, _ = count_crossers([det("person_left", (0, 0, 50, 150)), det("person_left", (30, 0, 80, 150))])
    assert counts["person_left"] == 2  # IoU 0.25: two people side by side


def test_companion_inside_a_wheeled_box_is_kept():
    counts, _ = count_crossers([det("wheeled_right", (0, 0, 200, 200)), det("person_right", (20, 10, 80, 190))])
    assert counts["wheeled_right"] == 1 and counts["person_right"] == 1
