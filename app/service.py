"""The service logic behind the Gradio page, kept free of Gradio so it can be tested.

Extends the Lesson 4 notebook: same function names and reply shape
(validate_image, count_people, build_reply, annotate, predict), with the
counting step swapped for src/counting.py so the reply carries pedestrian and
cyclist counts per direction.
"""

import io
import logging
import os
import time

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from src.config import CONF_THRESHOLD
from src.counting import count_crossers

SERVICE_VERSION = "2.0.0"
MAX_IMAGE_PIXELS = 40_000_000       # same limit as the L4 notebook
MAX_FILE_BYTES = 15 * 1024 * 1024   # PRD §9

log = logging.getLogger("c389.service")

# Set once at startup by load_model(), never per request.
DETECTOR = None


def load_model(weights, bike_weights="yolo26n.pt"):
    """Load the detector once. Returns it, and keeps it for count_people()."""
    global DETECTOR
    from src.detector import Detector  # imported here so tests can run without ultralytics
    DETECTOR = Detector(weights, bike_weights)
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

    return image.convert("RGB"), None


def count_people(image, threshold, detector=None):
    """Run the detector on one validated image, then link and count.

    Returns (counts, detections). Every count key is present, and an image with
    nobody in it is a valid result of zeros rather than an error.
    """
    detector = detector or DETECTOR
    if detector is None:
        raise RuntimeError("call load_model() before count_people()")
    return count_crossers(detector(image), threshold)


def build_reply(uploaded, threshold=CONF_THRESHOLD, detector=None):
    """Validate, count, answer. Returns (reply, image); image is None on refusal."""
    started = time.time()
    image, error = validate_image(uploaded)
    if error is not None:
        _log_request(error["status"], None, started)
        return error, None

    detector = detector or DETECTOR
    counts, detections = count_people(image, threshold, detector)
    elapsed = time.time() - started

    reply = {
        "status": 200,
        "counts": counts,
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


# Colours: pedestrians blue, cyclists orange, bicycles green, rider-bike links yellow.
COLOURS = {
    "pedestrian": (46, 112, 196),
    "cyclist": (230, 126, 34),
    "bicycle": (39, 174, 96),
    "link": (241, 196, 15),
}
ARROWS = {"left": "<-", "right": "->"}


def annotate(image, detections):
    """Draw boxes by role, and a line from each cyclist to their bike."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    width = max(2, round(max(image.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(max(image.size) / 50)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    by_id = {d["id"]: d for d in detections}
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        if det["class_name"] == "bicycle":
            colour = COLOURS["bicycle"]
            label = "bicycle %.2f" % det["confidence"]
        else:
            colour = COLOURS[det["role"]]
            direction = det["class_name"].split("_")[1]
            label = "%s %s %.2f" % (det["role"], ARROWS[direction], det["confidence"])
            if det.get("linked_bicycle") is not None:
                bx1, by1, bx2, by2 = by_id[det["linked_bicycle"]]["box"]
                draw.line([((x1 + x2) / 2, (y1 + y2) / 2), ((bx1 + bx2) / 2, (by1 + by2) / 2)],
                          fill=COLOURS["link"], width=width)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        draw.text((x1 + 3, max(0, y1 - font.size - 4) if hasattr(font, "size") else y1),
                  label, fill=colour, font=font)
    return marked


def predict(uploaded, threshold=CONF_THRESHOLD):
    """What the page calls. Returns (annotated image or None, reply)."""
    reply, image = build_reply(uploaded, threshold)
    if image is None:
        return None, reply
    return annotate(image, reply["detections"]), reply
