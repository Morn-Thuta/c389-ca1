"""Tests for src/linking.py: which person rides which bike (hand-made boxes)."""

from src.linking import link_people_to_bikes, link_score, union_box

# A rider: person box 100-200 wide, 0-300 tall; bike under the legs.
RIDER = [100, 0, 200, 300]
BIKE_UNDER = [80, 180, 220, 300]
# A walker pushing a bike: the bike sits beside them, not under them.
WALKER = [300, 0, 380, 300]
BIKE_BESIDE = [390, 180, 530, 300]


def test_rider_links_to_bike_under_them():
    links = link_people_to_bikes([RIDER], [BIKE_UNDER])
    assert list(links) == [0] and links[0][0] == 0


def test_pushing_a_bike_does_not_link():
    assert link_people_to_bikes([WALKER], [BIKE_BESIDE]) == {}


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
    # Two people over the same bike: only the better overlap rides it.
    other = [110, 20, 190, 260]
    assert list(link_people_to_bikes([RIDER, other], [BIKE_UNDER])) == [0]


def test_two_riders_two_bikes():
    rider2 = [400, 0, 500, 300]
    bike2 = [380, 180, 520, 300]
    links = link_people_to_bikes([RIDER, rider2], [BIKE_UNDER, bike2])
    assert {p: b for p, (b, _) in links.items()} == {0: 0, 1: 1}


def test_union_box_holds_rider_and_bike():
    assert union_box(RIDER, BIKE_UNDER) == [80, 0, 220, 300]
