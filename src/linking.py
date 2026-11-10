"""Decide which person is with which bike, and merge the pair into one wheeled box.

The model never sees separate bikes: a person and their vehicle are labelled as
one wheeled_left / wheeled_right box. Team rule: riding, walking or carrying the
bike all count. This rule is used where a person box and a bike box arrive
separately: converting old exports (src/data/to_wheeled.py) and pre-labelling
with the stock COCO model (src/data/prelabel.py).

A bike is a person's when either
  under:  the bike sits under their lower body (riding)
    - lower   = the bottom LINK_LOWER_FRACTION of the person's box
    - overlap = area(lower ∩ bike) / area(bike)        must be >= LINK_MIN_OVERLAP
    - x_ok    = the person's x-centre lies within the bike's x-range,
                widened by LINK_X_EXPAND of the bike's width on each side
    or, when the person box stops at the saddle: x_ok, the person starts above
    the bike, and the bottom of the person box falls within the bike's height
  beside: the bike is level with their lower body and next to it (walking it)
    - at least LINK_BESIDE_MIN_SPAN of the bike's height is level with `lower`
    - the horizontal gap between the boxes is at most LINK_BESIDE_GAP of the
      person's width (touching or overlapping is a gap of 0)
"Under" scores its overlap; "beside" scores below LINK_MIN_OVERLAP, so a rider
always wins a bike over someone standing next to it. Candidates are matched
greedily, highest score first: each bike goes to at most one person and each
person to at most one bike. A bike with nobody near it (parked) stays unlinked.
"""

from src.config import (
    LINK_BESIDE_GAP,
    LINK_BESIDE_MIN_SPAN,
    LINK_LOWER_FRACTION,
    LINK_MIN_OVERLAP,
    LINK_X_EXPAND,
)


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
    """Return a score if the bike can be this person's, else None (see module doc)."""
    bike_area = area(bike_box)
    if bike_area == 0:
        return None
    lower = lower_body(person_box)
    overlap = area(intersection(lower, bike_box)) / bike_area

    centre_x = (person_box[0] + person_box[2]) / 2
    pad = (bike_box[2] - bike_box[0]) * LINK_X_EXPAND
    x_ok = bike_box[0] - pad <= centre_x <= bike_box[2] + pad
    if overlap >= LINK_MIN_OVERLAP and x_ok:
        return overlap
    # Seated: the person's box stops at the saddle (legs drawn inside the bike box),
    # so the overlap is small but their box ends within the bike's height.
    seated = bike_box[1] <= person_box[3] <= bike_box[3] and person_box[1] < bike_box[1]
    if seated and x_ok:
        return LINK_MIN_OVERLAP

    level = max(0.0, min(lower[3], bike_box[3]) - max(lower[1], bike_box[1]))
    gap = max(0.0, max(person_box[0], bike_box[0]) - min(person_box[2], bike_box[2]))
    max_gap = (person_box[2] - person_box[0]) * LINK_BESIDE_GAP
    if level >= LINK_BESIDE_MIN_SPAN * (bike_box[3] - bike_box[1]) and gap <= max_gap:
        closeness = 1 - gap / max_gap if max_gap > 0 else 1
        return LINK_MIN_OVERLAP * 0.5 * (1 + closeness) * 0.99  # always below an "under" match
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
