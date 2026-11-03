"""Count crossers per image by class and direction (PRD §7).

Input is the detection list the L3/L4 notebooks produce:
    {"class_name": str, "confidence": float, "box": [x1, y1, x2, y2]}

The model's four classes are the answer: person_left / person_right for people
on foot, wheeled_left / wheeled_right for a rider and their bike or scooter in
one box. Counting is therefore a threshold and a tally; the totals add both
kinds per direction.

The threshold lives in src/config.py and is tuned on the validation set only.
"""

from src.config import CLASS_NAMES, CONF_THRESHOLD

# Every reply carries all of these, even at 0.
COUNT_KEYS = CLASS_NAMES + ["total_left", "total_right"]


def direction_of(class_name):
    """'person_left' -> 'left', 'wheeled_right' -> 'right'."""
    return class_name.rsplit("_", 1)[1]


def count_crossers(detections, threshold=CONF_THRESHOLD):
    """Drop detections under the threshold or outside our classes, then count.

    Returns (counts, kept): counts has every key in COUNT_KEYS, and kept is the
    surviving detections, each given an "id" and its "direction".
    """
    kept = []
    for det in detections:
        if det["confidence"] < threshold or det["class_name"] not in CLASS_NAMES:
            continue
        kept.append(dict(det, id=len(kept), direction=direction_of(det["class_name"])))

    counts = dict.fromkeys(COUNT_KEYS, 0)
    for det in kept:
        counts[det["class_name"]] += 1
        counts["total_" + det["direction"]] += 1
    return counts, kept
