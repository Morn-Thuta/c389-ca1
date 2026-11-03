"""Decide which person rides which bike, and merge the pair into one wheeled box.

The model never sees separate bikes: a rider and their vehicle are labelled as
one wheeled_left / wheeled_right box. This rule is used where a person box and
a bike box arrive separately: converting old exports (src/data/to_wheeled.py)
and pre-labelling with the stock COCO model (src/data/prelabel.py).

A person rides a bike when the bike sits under their lower body:
  - lower   = the bottom LINK_LOWER_FRACTION of the person's box
  - overlap = area(lower ∩ bike) / area(bike)          must be >= LINK_MIN_OVERLAP
  - x_ok    = the person's x-centre lies within the bike's x-range,
              widened by LINK_X_EXPAND of the bike's width on each side
Candidates are matched greedily, highest overlap first, so each bike links to at
most one person and each person to at most one bike. Someone pushing a bike walks
beside it, so the bike is not under them and they stay a pedestrian.
"""

from src.config import LINK_LOWER_FRACTION, LINK_MIN_OVERLAP, LINK_X_EXPAND


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a, b):
    """The overlapping box of a and b (may have zero area)."""
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def union_box(a, b):
    """The smallest box holding both a and b: the rider plus their vehicle."""
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def lower_body(box, fraction=LINK_LOWER_FRACTION):
    x1, y1, x2, y2 = box
    return [x1, y2 - (y2 - y1) * fraction, x2, y2]


def link_score(person_box, bike_box):
    """Return the overlap score if the bike can be this person's, else None."""
    bike_area = area(bike_box)
    if bike_area == 0:
        return None
    overlap = area(intersection(lower_body(person_box), bike_box)) / bike_area

    centre_x = (person_box[0] + person_box[2]) / 2
    pad = (bike_box[2] - bike_box[0]) * LINK_X_EXPAND
    x_ok = bike_box[0] - pad <= centre_x <= bike_box[2] + pad

    if overlap >= LINK_MIN_OVERLAP and x_ok:
        return overlap
    return None


def link_people_to_bikes(people, bikes):
    """Greedy one-to-one matching. people/bikes are lists of boxes.

    Returns {person_index: (bike_index, overlap)}.
    """
    candidates = []
    for p, person_box in enumerate(people):
        for b, bike_box in enumerate(bikes):
            score = link_score(person_box, bike_box)
            if score is not None:
                candidates.append((score, p, b))
    candidates.sort(key=lambda c: -c[0])

    links, used_bikes = {}, set()
    for score, p, b in candidates:
        if p in links or b in used_bikes:
            continue
        links[p] = (b, score)
        used_bikes.add(b)
    return links
