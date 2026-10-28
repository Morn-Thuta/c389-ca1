"""Run the YOLO model(s) and return detections in the L3/L4 notebook shape.

Two modes:
  - one model whose classes include `bicycle` (the final `twist_3cls`)
  - stop-gap until `twist_3cls` exists: the 2-class baseline for person_left /
    person_right, plus a stock COCO model (`yolo26n.pt`) for `bicycle` only

Either way the caller gets one list of {class_name, confidence, box} dicts, so
src/counting.py and the app never need to know which mode is running.
"""

from ultralytics import YOLO

from src.config import PERSON_CLASSES

BICYCLE = "bicycle"
# Keep everything the models find; the confidence threshold is applied later in
# counting.py, so the slider in the app can move without re-running the model.
RAW_CONF = 0.01


def to_detections(result, keep):
    """Turn one ultralytics result into detection dicts, keeping only class names in `keep`."""
    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections
    for i in range(len(boxes)):
        name = result.names[int(boxes.cls[i].item())]
        if name not in keep:
            continue
        detections.append({
            "class_name": name,
            "confidence": round(float(boxes.conf[i].item()), 3),
            "box": [round(float(v), 1) for v in boxes.xyxy[i].tolist()],
        })
    return detections


class Detector:
    def __init__(self, weights, bike_weights="yolo26n.pt"):
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.class_names = [self.model.names[k] for k in sorted(self.model.names)]

        # A model that already knows bicycles needs no helper.
        self.bike_weights = None
        self.bike_model = None
        self.bike_class_id = None
        if BICYCLE not in self.class_names and bike_weights:
            self.bike_weights = str(bike_weights)
            self.bike_model = YOLO(self.bike_weights)
            ids = [k for k, v in self.bike_model.names.items() if v == BICYCLE]
            if not ids:
                raise ValueError(f"{bike_weights} has no '{BICYCLE}' class")
            self.bike_class_id = ids[0]

    @property
    def mode(self):
        return "single model" if self.bike_model is None else "baseline + stock bicycle model"

    def __call__(self, image):
        result = self.model.predict(image, conf=RAW_CONF, verbose=False)[0]
        detections = to_detections(result, PERSON_CLASSES | {BICYCLE})
        if self.bike_model is not None:
            result = self.bike_model.predict(
                image, conf=RAW_CONF, classes=[self.bike_class_id], verbose=False)[0]
            detections += to_detections(result, {BICYCLE})
        return detections
