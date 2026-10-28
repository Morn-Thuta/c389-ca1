"""Gradio page for the People Counting Assistant + Twist 2 (cyclists).

Colab:
    !python -m app.app --share
Local:
    python -m app.app --weights path/to/best.pt

Weights default to $PROJECT_ROOT/artifacts/L3/best.pt (the baseline from the L3
notebook), with yolo26n.pt supplying bicycles until twist_3cls is trained. When
the weights already include `bicycle`, the helper model is not loaded.
"""

import argparse
import logging
import os

import gradio as gr

from app import service
from src.config import ARTIFACTS_DIR, CONF_THRESHOLD

DEFAULT_WEIGHTS = os.environ.get("MODEL_WEIGHTS", str(ARTIFACTS_DIR / "L3" / "best.pt"))


def build_demo():
    return gr.Interface(
        fn=service.predict,
        inputs=[
            gr.File(label="Crossing photograph (JPEG or PNG)", type="filepath"),
            gr.Slider(0.05, 0.95, value=CONF_THRESHOLD, step=0.05, label="Confidence threshold"),
        ],
        outputs=[
            gr.Image(label="Detections (red = moving left, blue = moving right, thick box + yellow line = cyclist, green = bicycle)"),
            gr.JSON(label="Reply"),
        ],
        title="People Counting Assistant: pedestrians and cyclists",
        description="Upload one photograph of the crossing. The reply counts pedestrians "
                    "and cyclists moving left and right across the image.",
        flagging_mode="never",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="person model weights (default: %(default)s)")
    parser.add_argument("--bike-weights", default="yolo26n.pt",
                        help="stock model used for bicycles when --weights has no bicycle class; 'none' to disable")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not os.path.exists(args.weights):
        parser.error(f"weights not found: {args.weights} (set PROJECT_ROOT or MODEL_WEIGHTS, or pass --weights)")
    bike = None if args.bike_weights.lower() == "none" else args.bike_weights
    service.load_model(args.weights, bike)

    demo = build_demo()
    try:
        demo.launch(share=args.share)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
