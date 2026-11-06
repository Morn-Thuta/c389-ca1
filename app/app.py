"""Gradio page for the People Counting Assistant + Twist 2 (micromobility crossers).

Colab:
    !python -m app.app --share
Local:
    python -m app.app --weights path/to/best.pt

Weights default to $PROJECT_ROOT/artifacts/L3/best.pt (the baseline from the L3
notebook, people only). Point --weights at the twist_4cls best.pt to count
wheeled crossers too.
"""

import argparse
import logging
import os

import gradio as gr

from app import service
from src.config import ARTIFACTS_DIR, CONF_THRESHOLD, INPUT_STRETCH

DEFAULT_WEIGHTS = os.environ.get("MODEL_WEIGHTS", str(ARTIFACTS_DIR / "L3" / "best.pt"))


def build_demo():
    return gr.Interface(
        fn=service.predict,
        inputs=[
            gr.File(label="Crossing photograph (JPEG or PNG)", type="filepath"),
            gr.Slider(0.05, 0.95, value=CONF_THRESHOLD, step=0.05, label="Confidence threshold"),
        ],
        outputs=[
            gr.Image(label="Detections (people: red = left, blue = right; wheeled, thick box: orange = left, purple = right)"),
            gr.JSON(label="Reply"),
        ],
        title="People Counting Assistant: people and wheeled crossers",
        description="Upload one photograph of the crossing. The reply counts people on foot "
                    "and wheeled crossers (bikes, scooters) moving left and right across the image.",
        flagging_mode="never",
    )


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
    service.load_model(args.weights, args.stretch or None)

    demo = build_demo()
    try:
        demo.launch(share=args.share)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
