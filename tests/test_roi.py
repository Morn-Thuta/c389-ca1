"""Tests for src/roi.py and the ROI step in count_crossers (no model, no images)."""

import pytest

from src.counting import count_crossers
from src.roi import anchor_of, in_roi, parse_roi, point_in_polygon

# Left half of a 200 x 100 image.
LEFT_HALF = [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]
SIZE = (200, 100)


def det(name, box, conf=0.9):
    return {"class_name": name, "confidence": conf, "box": list(box)}


def test_parse_roi_accepts_points_and_nothing():
    assert parse_roi(None) is None
    assert parse_roi([]) is None
    assert parse_roi([[0, 0], [1, 0], [0.5, 1]]) == [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]


@pytest.mark.parametrize("roi, message", [
    ([[0, 0], [1, 1]], "at least 3"),
    ([[0, 0], [1, 0], [2, 1]], "0 to 1"),
    ([[0, 0], [1, 0], "a"], "[x, y]"),
    ([[0, 0], [1, 0], ["a", 1]], "numbers"),
    ("not a list", "list"),
])
def test_parse_roi_refuses_bad_input(roi, message):
    with pytest.raises(ValueError, match=message):
        parse_roi(roi)


def test_anchor_is_bottom_centre():
    assert anchor_of([20, 10, 60, 90], SIZE) == (0.2, 0.9)


def test_point_in_polygon_including_concave_shape():
    assert point_in_polygon((0.25, 0.5), LEFT_HALF)
    assert not point_in_polygon((0.75, 0.5), LEFT_HALF)
    l_shape = [(0, 0), (1, 0), (1, 0.3), (0.3, 0.3), (0.3, 1), (0, 1)]
    assert point_in_polygon((0.1, 0.9), l_shape)
    assert not point_in_polygon((0.8, 0.8), l_shape)


def test_no_roi_means_inside():
    assert in_roi([150, 0, 190, 100], None, SIZE)


def test_counts_only_feet_inside_the_area():
    dets = [
        det("person_left", (10, 10, 50, 90)),     # feet at x=0.15: inside
        det("person_right", (120, 10, 160, 90)),  # feet at x=0.70: outside
        det("wheeled_right", (40, 0, 80, 100)),   # feet at x=0.30: inside
        det("person_left", (90, 0, 130, 50)),     # body crosses the line, feet at x=0.55: outside
    ]
    counts, kept = count_crossers(dets, 0.4, roi=LEFT_HALF, image_size=SIZE)
    assert counts["person_left"] == 1
    assert counts["person_right"] == 0
    assert counts["wheeled_right"] == 1
    assert counts["total_left"] == 1 and counts["total_right"] == 1
    assert len(kept) == 4                          # everything is still returned
    assert [d["in_roi"] for d in kept] == [True, False, True, False]  # equal confidence keeps input order


def test_duplicates_merge_before_the_roi():
    # One crosser on the edge, boxed twice: one copy's feet inside, the other's outside.
    dets = [det("person_left", (60, 0, 98, 100), 0.9), det("person_left", (62, 0, 104, 100), 0.6)]
    counts, kept = count_crossers(dets, 0.4, roi=LEFT_HALF, image_size=SIZE)
    assert len(kept) == 1 and counts["person_left"] == 1


def test_roi_needs_image_size():
    with pytest.raises(ValueError, match="image_size"):
        count_crossers([det("person_left", (0, 0, 10, 10))], roi=LEFT_HALF)
