"""Count crossers per image and link each person to the bicycle they ride (PRD §7).

Input is the detection list the L3/L4 notebooks produce:
    {"class_name": str, "confidence": float, "box": [x1, y1, x2, y2]}

A person is a cyclist when a bicycle sits under their lower body:
  - lower   = the bottom LINK_LOWER_FRACTION of the person's box
  - overlap = area(lower ∩ bike) / area(bike)          must be >= LINK_MIN_OVERLAP
  - x_ok    = the person's x-centre lies within the bike's x-range,
              widened by LINK_X_EXPAND of the bike's width on each side
Candidates are matched greedily, highest overlap first, so each bike links to at
most one person and each person to at most one bike. Someone pushing a bike walks
beside it, so the bike is not under them and they stay a pedestrian.

The thresholds live in src/config.py and are tuned on the validation set only.
"""

from src.config import (
    CONF_THRESHOLD,
    LINK_LOWER_FRACTION,
    LINK_MIN_OVERLAP,
    LINK_X_EXPAND,
    PERSON_CLASSES,
)

BICYCLE = "bicycle"

# Every reply carries all of these, even at 0.
COUNT_KEYS = [
    "pedestrian_left", "pedestrian_right",
    "cyclist_left", "cyclist_right",
    "bicycle_total", "bicycle_unridden",
    "person_left", "person_right",
]


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a, b):
    """The overlapping box of a and b (may have zero area)."""
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


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


def count_crossers(detections, threshold=CONF_THRESHOLD):
    """Drop detections under the threshold, link people to bikes and count.

    Returns (counts, kept): counts has every key in COUNT_KEYS, and kept is the
    surviving detections, each given an "id". People also get "role" and
    "linked_bicycle"; bicycles get "linked_person" (None when unridden).
    """
    kept = []
    for det in detections:
        if det["confidence"] < threshold:
            continue
        if det["class_name"] not in PERSON_CLASSES and det["class_name"] != BICYCLE:
            continue
        kept.append(dict(det, id=len(kept)))

    people = [d for d in kept if d["class_name"] in PERSON_CLASSES]
    bikes = [d for d in kept if d["class_name"] == BICYCLE]
    links = link_people_to_bikes([d["box"] for d in people], [d["box"] for d in bikes])

    for bike in bikes:
        bike["linked_person"] = None
    for p, person in enumerate(people):
        if p in links:
            bike = bikes[links[p][0]]
            person["role"] = "cyclist"
            person["linked_bicycle"] = bike["id"]
            bike["linked_person"] = person["id"]
        else:
            person["role"] = "pedestrian"
            person["linked_bicycle"] = None

    counts = dict.fromkeys(COUNT_KEYS, 0)
    for person in people:
        direction = person["class_name"].split("_")[1]  # "left" or "right"
        counts[f"{person['role']}_{direction}"] += 1
        counts[person["class_name"]] += 1
    counts["bicycle_total"] = len(bikes)
    counts["bicycle_unridden"] = sum(1 for b in bikes if b["linked_person"] is None)

    return counts, kept
