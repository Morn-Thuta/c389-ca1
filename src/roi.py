"""Region of interest (ROI): count only crossers inside the marked crossing area.

The teacher allowed this on 2 Nov: the detector still finds every visible person,
and the ROI is a counting step on top. A detection is inside when its anchor
point, the bottom centre of the box (where the feet or wheels meet the ground),
falls inside the ROI polygon. The bottom centre is used rather than the box
centre so a tall person standing just behind the crossing, whose body overlaps
it in the photo, is not counted.

An ROI is a list of at least 3 [x, y] points, each a fraction of the image width
and height (0 to 1), so one ROI fits the same camera view at any photo size.
"""

MIN_POINTS = 3


def parse_roi(roi):
    """Check an ROI and return it as a list of (x, y) floats, or None for no ROI.

    Raises ValueError with a message saying what is wrong.
    """
    if roi is None or (isinstance(roi, (list, tuple)) and len(roi) == 0):
        return None
    if not isinstance(roi, (list, tuple)):
        raise ValueError("The ROI must be a list of [x, y] points.")
    if len(roi) < MIN_POINTS:
        raise ValueError("The ROI needs at least %d points; got %d." % (MIN_POINTS, len(roi)))
    points = []
    for p in roi:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("Each ROI point must be [x, y]; got %r." % (p,))
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            raise ValueError("ROI points must be numbers; got %r." % (p,)) from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("ROI points are fractions of the image (0 to 1); got %r." % (p,))
        points.append((x, y))
    return points


def anchor_of(box, image_size):
    """Bottom centre of a pixel box, as fractions of the image size.

    Kept just inside the image: a person cut off by the bottom of the photo has a box
    ending exactly on the edge, and an area drawn down to the edge should include them.
    """
    x1, _, x2, y2 = box
    width, height = image_size
    inner = 1 - 1e-6
    return (min(max((x1 + x2) / 2 / width, 0.0), inner), min(max(y2 / height, 0.0), inner))


def point_in_polygon(point, polygon):
    """Ray casting: True when point is inside polygon (points on an edge may go either way)."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def in_roi(box, roi, image_size):
    """True when the box's anchor point is inside the ROI (always True with no ROI)."""
    if not roi:
        return True
    return point_in_polygon(anchor_of(box, image_size), roi)
