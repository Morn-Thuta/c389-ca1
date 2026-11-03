"""Run the YOLO model and return detections in the L3/L4 notebook shape.

Works with either model the project trains:
  - the 2-class baseline (person_left, person_right)
  - the Twist 2 model (person_left, person_right, wheeled_left, wheeled_right)

Either way the caller gets one list of {class_name, confidence, box} dicts, so
src/counting.py and the app never need to know which one is loaded.
"""

from ultralytics import YOLO

from src.config import CLASS_NAMES, WHEELED_CLASSES

# Keep everything the model finds; the confidence threshold is applied later in
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
    def __init__(self, weights):
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.class_names = [self.model.names[k] for k in sorted(self.model.names)]
        if self.class_names[:2] != CLASS_NAMES[:2]:
            raise ValueError(f"{weights} classes {self.class_names} must start with {CLASS_NAMES[:2]}")

    @property
    def mode(self):
        if WHEELED_CLASSES <= set(self.class_names):
            return "twist (people + wheeled)"
        return "baseline (people only)"

    def __call__(self, image):
        result = self.model.predict(image, conf=RAW_CONF, verbose=False)[0]
        return to_detections(result, set(CLASS_NAMES))
