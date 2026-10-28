"""Tests for src/counting.py, using hand-made boxes (no model, no images)."""

from src.counting import COUNT_KEYS, count_crossers, link_score


def det(name, box, conf=0.9):
    return {"class_name": name, "confidence": conf, "box": box}


# A rider: person box 100-200 wide, 0-300 tall; bike under the legs.
RIDER = [100, 0, 200, 300]
BIKE_UNDER = [80, 180, 220, 300]
# A walker pushing a bike: the bike sits beside them, not under them.
WALKER = [300, 0, 380, 300]
BIKE_BESIDE = [390, 180, 530, 300]


def test_empty_scene_has_every_key_at_zero():
    counts, kept = count_crossers([])
    assert list(counts) == COUNT_KEYS
    assert all(v == 0 for v in counts.values())
    assert kept == []


def test_rider_is_cyclist():
    counts, kept = count_crossers([det("person_right", RIDER), det("bicycle", BIKE_UNDER)])
    assert counts["cyclist_right"] == 1
    assert counts["pedestrian_right"] == 0
    assert counts["person_right"] == 1
    assert counts["bicycle_total"] == 1
    assert counts["bicycle_unridden"] == 0
    person, bike = kept
    assert person["role"] == "cyclist"
    assert person["linked_bicycle"] == bike["id"]
    assert bike["linked_person"] == person["id"]


def test_pushing_a_bike_is_pedestrian():
    counts, _ = count_crossers([det("person_left", WALKER), det("bicycle", BIKE_BESIDE)])
    assert counts["pedestrian_left"] == 1
    assert counts["cyclist_left"] == 0
    assert counts["bicycle_unridden"] == 1


def test_bike_only_overlapping_head_does_not_link():
    bike_at_head = [100, 0, 200, 80]
    assert link_score(RIDER, bike_at_head) is None


def test_x_centre_outside_bike_does_not_link():
    # Overlap is high (0.6) but the person's centre (x=180) is left of the bike's
    # widened range (190-310), so it is not theirs.
    wide_person = [100, 0, 260, 300]
    shifted = [200, 180, 300, 300]
    assert link_score(wide_person, shifted) is None


def test_one_bike_links_to_one_person_only():
    # Two people over the same bike: only the better overlap becomes a cyclist.
    other = [110, 20, 190, 260]
    counts, kept = count_crossers([
        det("person_left", RIDER), det("person_left", other), det("bicycle", BIKE_UNDER),
    ])
    assert counts["cyclist_left"] == 1
    assert counts["pedestrian_left"] == 1
    assert kept[0]["role"] == "cyclist"  # RIDER covers more of the bike


def test_two_riders_two_bikes():
    rider2 = [400, 0, 500, 300]
    bike2 = [380, 180, 520, 300]
    counts, _ = count_crossers([
        det("person_left", RIDER), det("bicycle", BIKE_UNDER),
        det("person_right", rider2), det("bicycle", bike2),
    ])
    assert counts["cyclist_left"] == 1
    assert counts["cyclist_right"] == 1
    assert counts["bicycle_unridden"] == 0


def test_threshold_drops_low_confidence():
    counts, kept = count_crossers(
        [det("person_right", RIDER, 0.9), det("bicycle", BIKE_UNDER, 0.2)], threshold=0.4)
    assert counts["pedestrian_right"] == 1
    assert counts["bicycle_total"] == 0
    assert len(kept) == 1


def test_unknown_classes_are_ignored():
    counts, kept = count_crossers([det("car", [0, 0, 10, 10])])
    assert kept == []
    assert sum(counts.values()) == 0
