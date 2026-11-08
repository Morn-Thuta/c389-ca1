"""Gradio page for the People Counting Assistant + Twist 2 (micromobility crossers).

Input: any uploaded file (refused with a reason unless it is a readable photo) and the threshold.
Output: the photo with its boxes, the four counts and the JSON reply.
Crossing area (ROI): click the corners of the crossing on the Output photo. From the
third corner on, only crossers whose feet are inside the area are counted.

Colab:
    !python -m app.app --share
Local:
    python -m app.app --weights path/to/best.pt

Weights default to $PROJECT_ROOT/artifacts/L3/best.pt. Point --weights (or
MODEL_WEIGHTS) at the 4-class weights to count wheeled crossers too.

From a notebook, start it with launch(build_demo(), share=True) so the page gets
its theme and stylesheet (Gradio 6 takes those at launch).
"""

import argparse
import hashlib
import html
import logging
import os

import gradio as gr
from PIL import ImageDraw

from app import service
from src.config import ARTIFACTS_DIR, CLASS_NAMES, CONF_THRESHOLD, INPUT_STRETCH

DEFAULT_WEIGHTS = os.environ.get("MODEL_WEIGHTS", str(ARTIFACTS_DIR / "L3" / "best.pt"))

WHAT = {
    "person_left": "On foot · moving left",
    "person_right": "On foot · moving right",
    "wheeled_left": "Bike or scooter · moving left",
    "wheeled_right": "Bike or scooter · moving right",
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


# ---------------------------------------------------------------- look

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"],
    font_mono=["JetBrains Mono", "Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
).set(
    body_background_fill="#f5f6f8",
    body_background_fill_dark="#0e1116",
    block_background_fill="#ffffff",
    block_background_fill_dark="#161a21",
    block_border_color="#e3e6ea",
    block_border_color_dark="#262c35",
    block_shadow="none",
    block_label_text_weight="500",
    block_title_text_weight="500",
    button_primary_background_fill="#1f2937",
    button_primary_background_fill_hover="#111827",
    button_primary_background_fill_dark="#e5e7eb",
    button_primary_text_color_dark="#111827",
    slider_color="#1f2937",
    slider_color_dark="#e5e7eb",
)

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding-top: 8px !important; }
footer { display: none !important; }

.pc-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap;
             padding: 18px 2px 14px; border-bottom: 1px solid var(--block-border-color); margin-bottom: 4px; }
.pc-header h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.015em; color: var(--body-text-color); }
.pc-header p { margin: 4px 0 0; font-size: 14px; color: var(--body-text-color-subdued); max-width: 640px; }
.pc-chip { font-family: var(--font-mono); font-size: 12px; color: var(--body-text-color-subdued); border: 1px solid var(--block-border-color);
           border-radius: 999px; padding: 4px 10px; background: var(--block-background-fill); white-space: nowrap; }
.pc-chip i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #22a06b; margin-right: 6px; }

.pc-section { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
              color: var(--body-text-color-subdued); margin: 2px 2px 0; }

.pc-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.pc-stats > .pc-stat { background: var(--block-background-fill); border: 1px solid var(--block-border-color);
           border-radius: 10px; padding: 14px 16px 12px; min-width: 0; }
.pc-stat > .k { display: flex; align-items: center; gap: 8px; font-family: var(--font-mono); font-size: 13px;
                font-weight: 500; color: var(--body-text-color); }
.pc-stat > .k > i { width: 9px; height: 9px; border-radius: 3px; flex: none; }
.pc-stat > .n { font-size: 36px; line-height: 1.15; font-weight: 600; letter-spacing: -0.02em;
                font-variant-numeric: tabular-nums; color: var(--body-text-color); margin-top: 6px; }
.pc-stat > .d { font-size: 12.5px; color: var(--body-text-color-subdued); }
.pc-totals { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 10px;
             background: var(--block-background-fill); border: 1px solid var(--block-border-color); border-radius: 10px; }
.pc-totals > div { padding: 10px 14px; min-width: 0; }
.pc-totals > div + div { border-left: 1px solid var(--block-border-color); }
.pc-totals .k { font-family: var(--font-mono); font-size: 12px; color: var(--body-text-color-subdued);
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pc-totals .n { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--body-text-color); }
.pc-note { margin: 10px 2px 0; font-size: 13px; color: var(--body-text-color-subdued); line-height: 1.5; }
.pc-error { margin-top: 10px; border: 1px solid #f1c7c2; background: #fdf3f2; color: #8a2a1f;
            border-radius: 10px; padding: 10px 14px; font-size: 13.5px; line-height: 1.5; }
.dark .pc-error { border-color: #5c2a24; background: #2a1715; color: #f3b8b0; }
.pc-error b { font-weight: 600; }

.pc-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--body-text-color-subdued);
             padding: 2px 2px 0; }
.pc-legend span { display: inline-flex; align-items: center; gap: 6px; }
.pc-legend i { width: 12px; height: 12px; border-radius: 3px; border: 2px solid; }
.pc-legend i.fill { border-width: 0; }
.pc-legend i.dash { border-style: dashed; }

.pc-area { font-size: 13px; color: var(--body-text-color-subdued); line-height: 1.5; padding: 0 2px; }
.pc-area b { color: var(--body-text-color); font-weight: 600; }
.pc-steps { margin: 6px 0 0 18px; padding: 0; }
"""


def _rgb(colour):
    return "rgb(%d,%d,%d)" % colour


def header_html():
    detector = service.DETECTOR
    model = os.path.basename(getattr(detector, "weights", "") or "") or "no model loaded"
    return (
        '<div class="pc-header"><div>'
        "<h1>People Counting Assistant</h1>"
        "<p>Counts people on foot and wheeled crossers (a rider and their bike or scooter) moving left "
        "and right across a crossing photo.</p></div>"
        f'<span class="pc-chip"><i></i>{html.escape(model)} · v{service.SERVICE_VERSION}</span></div>'
    )


def legend_html():
    items = [(_rgb(service.COLOURS[k]), k, "") for k in CLASS_NAMES]
    items += [(_rgb(service.OUTSIDE_ROI), "outside the area (not counted)", "dash"),
              ("rgba(39,174,96,.25)", "crossing area", "fill")]
    chips = "".join(
        f'<span><i class="{cls}" style="border-color:{c};{"background:" + c if cls == "fill" else ""}"></i>{label}</span>'
        for c, label, cls in items)
    return f'<div class="pc-legend">{chips}</div>'


def counts_html(reply):
    """The four counts as cards, then the totals and a one-line note (or the refusal)."""
    ok = reply is not None and reply.get("status") == 200

    def value(key):
        return reply["counts"][key] if ok else "–"

    cards = "".join(
        f'<div class="pc-stat"><div class="k"><i style="background:{_rgb(service.COLOURS[k])}"></i>{k}:</div>'
        f'<div class="n">{value(k)}</div><div class="d">{WHAT[k]}</div></div>'
        for k in CLASS_NAMES)
    outside = reply["outside_roi"] if ok and reply.get("roi") else "–"
    totals = "".join(f'<div><div class="k">{k}:</div><div class="n">{v}</div></div>'
                     for k, v in [("total_left", value("total_left")), ("total_right", value("total_right")),
                                  ("outside_roi", outside)])
    if reply is None:
        note = '<p class="pc-note">Upload a photo to count.</p>'
    elif not ok:
        note = (f'<div class="pc-error"><b>Refused · {reply["status"]}</b> — {html.escape(reply["error"])} '
                f'{html.escape(reply.get("detail", ""))}</div>')
    elif reply.get("roi"):
        n = reply["outside_roi"]
        note = (f'<p class="pc-note">Counting inside the crossing area only. {n} detected outside it '
                f'{"is" if n == 1 else "are"} shown in grey and not counted.</p>')
    else:
        note = '<p class="pc-note">Counting everyone in the photo.</p>'
    return f'<div class="pc-stats">{cards}</div><div class="pc-totals">{totals}</div>{note}'


def area_note(corners):
    if not corners:
        return ('<div class="pc-area"><b>Crossing area: none.</b> Everyone in the photo is counted. '
                "To count only the crossing, click its corners on the output photo.</div>")
    if len(corners) < 3:
        more = 3 - len(corners)
        return (f'<div class="pc-area"><b>Crossing area: {len(corners)} of 3 corners.</b> '
                f'Click {more} more corner{"s" if more > 1 else ""} on the output photo.</div>')
    return (f'<div class="pc-area"><b>Crossing area: {len(corners)} corners.</b> Only crossers whose feet are '
            "inside are counted. More clicks add corners; the area is kept for the next photo.</div>")


# ---------------------------------------------------------------- behaviour

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
    """Count one uploaded file.

    Returns (input preview, output photo, count cards, JSON reply, reply for state).
    Any file reaches the service, so a text file, an empty file or a huge image is
    refused with its status and reason, the same as through the API.
    """
    if photo is None:
        return None, None, counts_html(None), None, None
    global CACHED
    if CACHED is None or CACHED.detector is not service.DETECTOR:  # model loaded with service.load_model()
        CACHED = CachedDetector(service.DETECTOR) if service.DETECTOR is not None else None
    roi = corners if len(corners) >= 3 else None
    reply, image = service.build_reply(photo, threshold, detector=CACHED, roi=roi)
    if image is None:
        gr.Warning(f"Refused ({reply['status']}): {reply['error']} {reply.get('detail', '')}")
        return None, None, counts_html(reply), reply, reply
    marked = service.annotate(image, reply["detections"], reply["roi"])
    if corners and roi is None:
        marked = draw_corners(marked, corners)
    return image, marked, counts_html(reply), reply, reply


def add_corner(photo, threshold, corners, last_reply, evt: gr.SelectData):
    if photo is None or not last_reply or last_reply.get("status") != 200:
        return (corners, *[gr.skip()] * 5, area_note(corners))
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
        gr.HTML(header_html())
        corners = gr.State([])
        last_reply = gr.State(None)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                gr.HTML('<div class="pc-section">Input</div>')
                photo = gr.File(label="Crossing photograph · JPEG or PNG, up to 15 MB", type="filepath", height=150)
                preview = gr.Image(label="Uploaded photo", type="pil", interactive=False, height=300,
                                   buttons=[])
                threshold = gr.Slider(0.05, 0.95, value=CONF_THRESHOLD, step=0.05, label="Confidence threshold",
                                      info="Detections below this score are ignored.")
            with gr.Column(scale=7, min_width=320):
                gr.HTML('<div class="pc-section">Output</div>')
                output = gr.Image(label="Detections · click the crossing's corners to set the area",
                                  type="pil", interactive=False, height=470, buttons=["download", "fullscreen"])
                gr.HTML(legend_html())
                area = gr.HTML(area_note([]))
                with gr.Row():
                    undo = gr.Button("Undo last corner", size="sm", variant="secondary")
                    clear = gr.Button("Clear area", size="sm", variant="secondary")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<div class="pc-section">Counts</div>')
                counts = gr.HTML(counts_html(None))
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<div class="pc-section">Reply</div>')
                reply_json = gr.JSON(label="JSON", open=True)

        outputs = [preview, output, counts, reply_json, last_reply]
        photo.change(run, [photo, threshold, corners], outputs)
        threshold.release(run, [photo, threshold, corners], outputs)
        output.select(add_corner, [photo, threshold, corners, last_reply], [corners, *outputs, area])
        undo.click(undo_corner, [photo, threshold, corners], [corners, *outputs, area])
        clear.click(clear_area, [photo, threshold], [corners, *outputs, area])
    return demo


def launch(demo, **kwargs):
    """Launch with the page's theme and stylesheet (Gradio 6 takes them at launch)."""
    return demo.launch(theme=THEME, css=CSS, **kwargs)


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
        launch(demo, share=args.share)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
