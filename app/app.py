"""Gradio page for the People Counting Assistant + Twist 2 (micromobility crossers).

Input: the crossing photo and the confidence threshold.
Output: the photo with its boxes, the four counts and the JSON reply.
Crossing area (ROI): click the corners of the crossing on the Output photo. From the
third corner on, only crossers whose feet are inside the area are counted.

Colab:
    !python -m app.app --share
Local:
    python -m app.app --weights path/to/best.pt

Weights default to $PROJECT_ROOT/artifacts/L3/best.pt. Point --weights (or
MODEL_WEIGHTS) at the 4-class weights to count wheeled crossers too.
"""

import argparse
import hashlib
import logging
import os

import gradio as gr
from PIL import ImageDraw

from app import service
from src.config import ARTIFACTS_DIR, CLASS_NAMES, CONF_THRESHOLD, INPUT_STRETCH

DEFAULT_WEIGHTS = os.environ.get("MODEL_WEIGHTS", str(ARTIFACTS_DIR / "L3" / "best.pt"))

WHAT = {
    "person_left": "on foot, moving left ←",
    "person_right": "on foot, moving right →",
    "wheeled_left": "wheeled, moving left ←",
    "wheeled_right": "wheeled, moving right →",
}


class CachedDetector:
    """Remembers the last photo's detections, so clicking area corners or moving the
    slider re-counts instantly instead of running the model again."""

    def __init__(self, detector):
        self.detector = detector
        self.mode = getattr(detector, "mode", "unknown")
        self._key = None
        self._detections = []

    def __call__(self, image):
        key = (image.size, hashlib.md5(image.tobytes()).hexdigest())
        if key != self._key:
            self._detections = self.detector(image)
            self._key = key
        return [dict(d) for d in self._detections]


CACHED = None


def counts_html(reply):
    """The four counts as cards, then the totals."""
    ok = reply is not None and reply.get("status") == 200
    def value(key):
        return reply["counts"][key] if ok else "–"

    cards = "".join(
        f'<div style="border:1px solid var(--block-border-color);border-left:6px solid rgb{service.COLOURS[k]};'
        f'border-radius:8px;padding:8px 12px">'
        f'<div style="font-family:monospace;font-weight:600">{k}:</div>'
        f'<div style="font-size:12px;opacity:.7">{WHAT[k]}</div>'
        f'<div style="font:700 30px monospace;color:rgb{service.COLOURS[k]}">{value(k)}</div></div>'
        for k in CLASS_NAMES)
    totals = [("total_left", value("total_left")), ("total_right", value("total_right")),
              ("outside_roi", reply["outside_roi"] if ok and reply.get("roi") else "–")]
    totals_html = "".join(
        f'<div style="background:var(--block-background-fill);border:1px solid var(--block-border-color);'
        f'border-radius:8px;padding:6px 10px"><div style="font:12px monospace;opacity:.7">{k}:</div>'
        f'<div style="font:700 20px monospace">{v}</div></div>' for k, v in totals)
    if not ok:
        note = reply["error"] if reply else "Upload a photo to count."
    elif reply.get("roi"):
        note = (f"Counting inside the crossing area only; {reply['outside_roi']} found outside it "
                f"(grey boxes) not counted.")
    else:
        note = "Counting everyone in the photo. Click the crossing's corners on the Output photo to count inside it only."
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">{cards}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px">{totals_html}</div>'
            f'<p style="margin:8px 0 0;font-size:13px;opacity:.8">{note}</p>')


def draw_corners(image, corners):
    """Show corners clicked so far (fewer than 3 is not an area yet)."""
    draw = ImageDraw.Draw(image)
    r = max(4, round(max(image.size) / 150))
    pts = [(x * image.width, y * image.height) for x, y in corners]
    if len(pts) == 2:
        draw.line(pts, fill=service.ROI_COLOUR, width=max(2, r // 2))
    for x, y in pts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=service.ROI_COLOUR, width=max(2, r // 2))
    return image


def run(photo, threshold, corners):
    """Count one photo. Returns (output photo, count cards, JSON reply, reply for state)."""
    if photo is None:
        return None, counts_html(None), None, None
    global CACHED
    if CACHED is None or CACHED.detector is not service.DETECTOR:  # model loaded with service.load_model()
        CACHED = CachedDetector(service.DETECTOR) if service.DETECTOR is not None else None
    roi = corners if len(corners) >= 3 else None
    reply, image = service.build_reply(photo, threshold, detector=CACHED, roi=roi)
    if image is None:
        return None, counts_html(reply), reply, reply
    marked = service.annotate(image, reply["detections"], reply["roi"])
    if corners and roi is None:
        marked = draw_corners(marked, corners)
    return marked, counts_html(reply), reply, reply


def area_note(corners):
    if not corners:
        return "**Crossing area:** none, everyone is counted. Click the crossing's corners on the Output photo to add one."
    if len(corners) < 3:
        return f"**Crossing area:** {len(corners)} corner(s); click {3 - len(corners)} more on the Output photo."
    return (f"**Crossing area:** {len(corners)} corners. Only crossers whose feet are inside are counted. "
            "Keep clicking to add corners; the area stays for the next photo.")


def add_corner(photo, threshold, corners, last_reply, evt: gr.SelectData):
    if photo is None or not last_reply or last_reply.get("status") != 200:
        return (corners, gr.skip(), gr.skip(), gr.skip(), gr.skip(), area_note(corners))
    width, height = last_reply["image_size"]
    x, y = evt.index
    corners = corners + [[round(min(max(x / width, 0), 1), 4), round(min(max(y / height, 0), 1), 4)]]
    return (corners, *run(photo, threshold, corners), area_note(corners))


def undo_corner(photo, threshold, corners):
    corners = corners[:-1]
    return (corners, *run(photo, threshold, corners), area_note(corners))


def clear_area(photo, threshold):
    return ([], *run(photo, threshold, []), area_note([]))


def build_demo():
    with gr.Blocks(title="People Counting Assistant") as demo:
        gr.Markdown("# People Counting Assistant: people and wheeled crossers\n"
                    "Upload one photograph of the crossing. The reply counts people on foot and wheeled "
                    "crossers (a bike or scooter with its rider) moving left and right.")
        corners = gr.State([])
        last_reply = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column():
                photo = gr.Image(label="Input: crossing photograph", type="filepath",
                                 sources=["upload", "clipboard"], height=420)
                threshold = gr.Slider(0.05, 0.95, value=CONF_THRESHOLD, step=0.05, label="Confidence threshold")
                area = gr.Markdown(area_note([]))
                with gr.Row():
                    undo = gr.Button("Undo last corner", size="sm")
                    clear = gr.Button("Clear crossing area", size="sm")
            with gr.Column():
                output = gr.Image(label="Output: detections (click the crossing's corners here to set the area)",
                                  type="pil", interactive=False, height=420)
                gr.Markdown("Boxes: <span style='color:rgb(214,69,65)'>■</span> person_left · "
                            "<span style='color:rgb(46,112,196)'>■</span> person_right · "
                            "<span style='color:rgb(230,126,34)'>■</span> wheeled_left (thick) · "
                            "<span style='color:rgb(142,68,173)'>■</span> wheeled_right (thick) · "
                            "<span style='color:rgb(150,150,150)'>■</span> outside the area, not counted · "
                            "<span style='color:rgb(39,174,96)'>■</span> crossing area")
        with gr.Row(equal_height=False):
            with gr.Column():
                counts = gr.HTML(counts_html(None), label="Counts")
            with gr.Column():
                reply_json = gr.JSON(label="Reply (JSON)")

        outputs = [output, counts, reply_json, last_reply]
        photo.change(run, [photo, threshold, corners], outputs)
        threshold.release(run, [photo, threshold, corners], outputs)
        output.select(add_corner, [photo, threshold, corners, last_reply], [corners, *outputs, area])
        undo.click(undo_corner, [photo, threshold, corners], [corners, *outputs, area])
        clear.click(clear_area, [photo, threshold], [corners, *outputs, area])
    return demo


def load(weights, stretch=INPUT_STRETCH):
    """Load the model once and wrap it so re-counting the same photo is instant."""
    global CACHED
    CACHED = CachedDetector(service.load_model(weights, stretch))
    return service.DETECTOR


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="model weights (default: %(default)s)")
    parser.add_argument("--stretch", type=int, default=INPUT_STRETCH,
                        help="squash photos to this square size first, matching the training export "
                             "(default: %(default)s; 0 for weights trained on unstretched photos)")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not os.path.exists(args.weights):
        parser.error(f"weights not found: {args.weights} (set PROJECT_ROOT or MODEL_WEIGHTS, or pass --weights)")
    load(args.weights, args.stretch or None)

    demo = build_demo()
    try:
        demo.launch(share=args.share)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
