"""The service logic behind the Gradio page, kept free of Gradio so it can be tested.

Extends the Lesson 4 notebook: same function names and reply shape
(validate_image, count_people, build_reply, annotate, predict), with the
counting step swapped for src/counting.py so the reply counts people on foot
and wheeled crossers (Twist 2) per direction.
"""

import io
import logging
import os
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from src.config import CONF_THRESHOLD, INPUT_STRETCH
from src.counting import count_crossers
from src.roi import parse_roi

SERVICE_VERSION = "2.1.0"
MAX_IMAGE_PIXELS = 40_000_000       # same limit as the L4 notebook
MAX_FILE_BYTES = 15 * 1024 * 1024   # PRD §9

log = logging.getLogger("c389.service")

# Set once at startup by load_model(), never per request.
DETECTOR = None


def load_model(weights, stretch=INPUT_STRETCH):
    """Load the detector once. Returns it, and keeps it for count_people().

    stretch: square size photos are squashed to before inference (the training
    export's "Stretch to" size), or None for weights trained on unstretched photos.
    """
    global DETECTOR
    from src.detector import Detector  # imported here so tests can run without ultralytics
    DETECTOR = Detector(weights, stretch)
    log.info("model loaded: %s (%s)", DETECTOR.weights, DETECTOR.mode)
    return DETECTOR


def _refuse(status, error, detail):
    return None, {"status": status, "error": error, "detail": detail}


def validate_image(uploaded):
    """Check that what arrived can be turned into an image to run on.

    Accepts a path, raw bytes, a PIL image, or None. Returns (image, error),
    exactly one of which is None. Refuses: nothing sent, an empty file, a file
    over 15 MB, a file that is not a readable image, and an image over the
    pixel limit. Every refusal says what was wrong and what to send instead.
    """
    if uploaded is None:
        return _refuse(400, "No image was sent.", "Attach a JPEG or PNG and send it again.")

    try:
        if isinstance(uploaded, Image.Image):
            image = uploaded
        elif isinstance(uploaded, (bytes, bytearray)):
            if not uploaded:
                return _refuse(400, "The upload was empty.", "The file carried no data.")
            if len(uploaded) > MAX_FILE_BYTES:
                return _refuse(413, "The file is too large.",
                               "Send an image under %d MB." % (MAX_FILE_BYTES // 2**20))
            image = Image.open(io.BytesIO(uploaded))
            image.load()
        else:
            size = os.path.getsize(uploaded)
            if size == 0:
                return _refuse(400, "The upload was empty.",
                               "The file '%s' is zero bytes." % os.path.basename(str(uploaded)))
            if size > MAX_FILE_BYTES:
                return _refuse(413, "The file is too large.",
                               "Send an image under %d MB." % (MAX_FILE_BYTES // 2**20))
            image = Image.open(uploaded)
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return _refuse(415, "That file is not a readable image.", "Send a JPEG or PNG.")

    if image.width * image.height > MAX_IMAGE_PIXELS:
        return _refuse(413, "The image is too large to process.",
                       "%d by %d pixels is beyond the limit of %d pixels."
                       % (image.width, image.height, MAX_IMAGE_PIXELS))

    # Phone photos are often stored sideways with an orientation tag; turn them upright
    # (as Roboflow did for the training photos, and as the browser shows them).
    return ImageOps.exif_transpose(image).convert("RGB"), None


def count_people(image, threshold, detector=None, roi=None):
    """Run the detector on one validated image, then count.

    roi: parsed ROI points, or None to count the whole image.
    Returns (counts, detections). Every count key is present, and an image with
    nobody in it is a valid result of zeros rather than an error.
    """
    detector = detector or DETECTOR
    if detector is None:
        raise RuntimeError("call load_model() before count_people()")
    return count_crossers(detector(image), threshold, roi=roi, image_size=image.size)


def validate_threshold(threshold):
    """Returns (threshold as float, error)."""
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return None, {"status": 422, "error": "The confidence threshold is not a number.",
                      "detail": "Send a number between 0.05 and 0.95."}
    if not 0.05 <= value <= 0.95:
        return None, {"status": 422, "error": "The confidence threshold is out of range.",
                      "detail": "Send a number between 0.05 and 0.95; got %s." % value}
    return value, None


def build_reply(uploaded, threshold=CONF_THRESHOLD, detector=None, roi=None):
    """Validate, count, answer. Returns (reply, image); image is None on refusal.

    roi: the crossing area as [[x, y], ...] fractions of the image (src/roi.py),
    or None to count everyone in the photo.
    """
    started = time.time()
    threshold, error = validate_threshold(threshold)
    if error is None:
        try:
            roi = parse_roi(roi)
        except ValueError as exc:
            error = {"status": 422, "error": "The crossing area (ROI) is not valid.", "detail": str(exc)}
    if error is None:
        image, error = validate_image(uploaded)
    if error is not None:
        _log_request(error["status"], None, started)
        return error, None

    detector = detector or DETECTOR
    counts, detections = count_people(image, threshold, detector, roi)
    elapsed = time.time() - started

    reply = {
        "status": 200,
        "counts": counts,
        "roi": [list(p) for p in roi] if roi else None,
        "outside_roi": sum(not d["in_roi"] for d in detections),
        "confidence_threshold": threshold,
        "model_version": SERVICE_VERSION,
        "model_mode": getattr(detector, "mode", "unknown"),
        "image_size": [image.width, image.height],
        "inference_seconds": round(elapsed, 3),
        "detections": detections,
    }
    _log_request(200, counts, started)
    return reply, image


def _log_request(status, counts, started):
    log.info("status=%d latency_ms=%.0f counts=%s",
             status, (time.time() - started) * 1000, counts)


# Box colour = class. People keep the L4 page's red (left) and blue (right); wheeled
# crossers get orange (left) and purple (right) with a thicker box.
COLOURS = {
    "person_left": (214, 69, 65),
    "person_right": (46, 112, 196),
    "wheeled_left": (230, 126, 34),
    "wheeled_right": (142, 68, 173),
}
ARROWS = {"left": "<-", "right": "->"}


OUTSIDE_ROI = (150, 150, 150)
ROI_COLOUR = (39, 174, 96)


def annotate(image, detections, roi=None):
    """Draw the ROI, then each detection in its class colour with a direction arrow and score.

    Detections outside the ROI are drawn thin and grey: found, but not counted.
    """
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    if roi:
        draw.polygon([(x * image.width, y * image.height) for x, y in roi],
                     outline=ROI_COLOUR, width=max(2, round(max(image.size) / 250)))
    width = max(2, round(max(image.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(max(image.size) / 45)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        kind, direction = det["class_name"].rsplit("_", 1)
        counted = det.get("in_roi", True)
        colour = COLOURS[det["class_name"]] if counted else OUTSIDE_ROI
        box_width = (width * 2 if kind == "wheeled" else width) if counted else max(1, width // 2)
        label = "%s %s %.2f" % (ARROWS[direction], "wheeled" if kind == "wheeled" else "walk",
                                det["confidence"])
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=box_width)

        # Label on a filled tab above the box, so it reads on any background.
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tab_h = bottom - top + 4
        ty = y1 - tab_h if y1 - tab_h >= 0 else y1
        draw.rectangle([x1, ty, x1 + right - left + 6, ty + tab_h], fill=colour)
        draw.text((x1 + 3, ty + 2 - top), label, fill=(255, 255, 255), font=font)
    return marked


def predict(uploaded, threshold=CONF_THRESHOLD, roi=None):
    """What the Gradio page calls. Returns (annotated image or None, reply)."""
    reply, image = build_reply(uploaded, threshold, roi=roi)
    if image is None:
        return None, reply
    return annotate(image, reply["detections"], reply["roi"]), reply
