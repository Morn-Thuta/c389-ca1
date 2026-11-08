"""Count crossers per image by class and direction (PRD §7).

Input is the detection list the L3/L4 notebooks produce:
    {"class_name": str, "confidence": float, "box": [x1, y1, x2, y2]}

The model's four classes are the answer: person_left / person_right for people
on foot, wheeled_left / wheeled_right for a rider and their bike or scooter in
one box. Counting is a threshold, a duplicate filter and a tally; the totals add
both kinds per direction.

YOLO26 is NMS-free and sometimes returns two near-identical boxes for one
crosser (seen: IoU 0.96 on a single cyclist), so boxes of the same kind
(person or wheeled, either direction) that overlap by DUPLICATE_IOU or more are
merged, keeping the more confident one.

With an ROI (src/roi.py), only crossers whose feet are inside the crossing area
are counted. Every detection is still returned, marked "in_roi", so the page can
show what was left out. Duplicates are merged before the ROI is applied, so one
crosser on the edge cannot be counted once inside and once outside.

The thresholds live in src/config.py and are tuned on the validation set only.
"""

from src.config import CLASS_NAMES, CONF_THRESHOLD, DUPLICATE_IOU
from src.roi import in_roi

# Every reply carries all of these, even at 0.
COUNT_KEYS = CLASS_NAMES + ["total_left", "total_right"]


def direction_of(class_name):
    """'person_left' -> 'left', 'wheeled_right' -> 'right'."""
    return class_name.rsplit("_", 1)[1]


def kind_of(class_name):
    """'person_left' -> 'person', 'wheeled_right' -> 'wheeled'."""
    return class_name.rsplit("_", 1)[0]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def count_crossers(detections, threshold=CONF_THRESHOLD, roi=None, image_size=None):
    """Drop detections under the threshold, outside our classes or duplicated, then count.

    roi: parsed ROI points (src/roi.parse_roi) or None to count the whole image;
    image_size (width, height) is needed with an ROI.

    Returns (counts, kept): counts has every key in COUNT_KEYS, and kept is the
    surviving detections, each given an "id", its "direction" and "in_roi".
    Only detections in the ROI are counted.
    """
    if roi and image_size is None:
        raise ValueError("image_size is needed to apply an ROI")
    candidates = [d for d in detections
                  if d["confidence"] >= threshold and d["class_name"] in CLASS_NAMES]
    kept = []
    for det in sorted(candidates, key=lambda d: -d["confidence"]):
        kind = kind_of(det["class_name"])
        if any(kind_of(k["class_name"]) == kind and iou(det["box"], k["box"]) >= DUPLICATE_IOU for k in kept):
            continue
        kept.append(dict(det, id=len(kept), direction=direction_of(det["class_name"]),
                         in_roi=in_roi(det["box"], roi, image_size)))

    counts = dict.fromkeys(COUNT_KEYS, 0)
    for det in kept:
        if not det["in_roi"]:
            continue
        counts[det["class_name"]] += 1
        counts["total_" + det["direction"]] += 1
    return counts, kept
