"""Our own web page for the People Counting Assistant + Twist 2, on a small JSON API.

The page (app/static/index.html) shows the photo with its boxes, the four counts as
cards and the full JSON reply side by side, and lets you draw the crossing area
(ROI). It calls the same service code as the Gradio page (app/service.py).

Local:
    python -m app.web --weights path/to/final_4cls_ig_v4_m.pt
    then open http://127.0.0.1:8000

API:
    GET  /api/health   model and settings
    POST /api/count    multipart form: image (file), threshold (0.05-0.95, optional),
                       roi (JSON [[x, y], ...] fractions of the image, optional)
                       -> the service reply, with its status as the HTTP status
"""

import argparse
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import service
from src.config import ARTIFACTS_DIR, CLASS_NAMES, CONF_THRESHOLD, INPUT_STRETCH
from src.counting import COUNT_KEYS

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_WEIGHTS = os.environ.get("MODEL_WEIGHTS", str(ARTIFACTS_DIR / "L3" / "best.pt"))

app = FastAPI(title="People Counting Assistant", version=service.SERVICE_VERSION,
              docs_url="/api/docs", redoc_url=None)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    detector = service.DETECTOR
    return {
        "status": 200 if detector is not None else 503,
        "model_loaded": detector is not None,
        "model_mode": getattr(detector, "mode", None),
        "model_weights": os.path.basename(getattr(detector, "weights", "") or "") or None,
        "classes": CLASS_NAMES,
        "count_keys": COUNT_KEYS,
        "default_threshold": CONF_THRESHOLD,
        "service_version": service.SERVICE_VERSION,
    }


@app.post("/api/count")
async def count(image: UploadFile | None = File(None),
                threshold: str = Form(str(CONF_THRESHOLD)),
                roi: str = Form("")):
    if service.DETECTOR is None:
        return JSONResponse({"status": 503, "error": "The model is not loaded.",
                             "detail": "Start the service with --weights."}, status_code=503)

    parsed_roi = None
    if roi.strip():
        try:
            parsed_roi = json.loads(roi)
        except json.JSONDecodeError:
            return JSONResponse({"status": 422, "error": "The crossing area (ROI) is not valid.",
                                 "detail": "Send it as JSON: [[x, y], ...] with x and y from 0 to 1."},
                                status_code=422)

    data = None
    if image is not None:
        # Read one byte past the limit, so an oversized file is refused without reading all of it.
        data = await image.read(service.MAX_FILE_BYTES + 1)
    reply, _ = service.build_reply(data, threshold, roi=parsed_roi)
    return JSONResponse(reply, status_code=reply["status"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="model weights (default: %(default)s)")
    parser.add_argument("--stretch", type=int, default=INPUT_STRETCH,
                        help="squash photos to this square size first, matching the training export "
                             "(default: %(default)s; 0 for weights trained on unstretched photos)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 = this computer only (default); 0.0.0.0 = others on the network")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not os.path.exists(args.weights):
        parser.error(f"weights not found: {args.weights} (set MODEL_WEIGHTS or pass --weights)")
    service.load_model(args.weights, args.stretch or None)

    import uvicorn
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
