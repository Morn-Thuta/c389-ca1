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
    counts, kept = count_crossers([
        det("person_left"), det("person_left"), det("person_right"),
        det("wheeled_left"), det("wheeled_right"), det("wheeled_right"),
    ])
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
