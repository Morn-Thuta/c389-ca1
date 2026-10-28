"""Pre-draw labels for every person and bicycle, for humans to correct in Roboflow.

  - Boxes come from a stock COCO model (yolo26n.pt): it finds people and bikes well,
    but cannot tell left from right.
  - Direction comes from our baseline (best.pt): each stock person box is matched
    to the baseline box that overlaps it most. If none overlaps enough, the person
    is cropped and the baseline is run again on the crop. If that also fails, the
    box is written as person_left and flagged "direction_unknown".
  - Existing (human) labels are kept as they are. A proposed box that overlaps an
    existing label is dropped, so only the missing people and bikes are added.

Output is a YOLO dataset in the 3-class order (person_left, person_right, bicycle)
that can be dropped into Roboflow's upload page, plus prelabel_review.csv listing
what was added to each image so reviewers can start with the busiest ones.

**Every image must still be checked by a person.** Use the original photos, not a
masked export: blurred people cannot be found.

Usage:
    python -m src.data.prelabel DATASET_DIR --out OUT_DIR --direction-weights best.pt
    python -m src.data.prelabel IMAGES_DIR  --out OUT_DIR --direction-weights best.pt
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from src.config import IMAGE_EXTS, SPLITS, TWIST_CLASSES

PERSON_LEFT, PERSON_RIGHT, BICYCLE = 0, 1, 2

DET_CONF = 0.25          # stock model: minimum confidence for a proposed box
DUPLICATE_IOU = 0.40     # proposal vs existing label: same object above this IoU
DUPLICATE_COVER = 0.70   # ...or when this much of the smaller box is covered
DIRECTION_IOU = 0.30     # stock person vs baseline box: same person above this IoU
CROP_PAD = 0.15          # padding around a person crop, as a fraction of box size


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cover(a, b):
    """Share of the smaller box covered by the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def is_duplicate(box, existing_boxes):
    return any(iou(box, e) >= DUPLICATE_IOU or cover(box, e) >= DUPLICATE_COVER
               for e in existing_boxes)


def match_direction(person_box, direction_dets):
    """Best-overlapping baseline detection -> (class_id, confidence), or None.

    direction_dets: [(class_id, confidence, box)] from the baseline on the full image.
    """
    best, best_iou = None, DIRECTION_IOU
    for class_id, conf, box in direction_dets:
        overlap = iou(person_box, box)
        if overlap >= best_iou:
            best, best_iou = (class_id, conf), overlap
    return best


# --- YOLO label files: "cls cx cy w h", normalised ---

def read_labels(path, width, height):
    """Return [(class_id, [x1, y1, x2, y2])] in pixels; [] if the file is missing."""
    labels = []
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        labels.append((c, [(cx - w / 2) * width, (cy - h / 2) * height,
                           (cx + w / 2) * width, (cy + h / 2) * height]))
    return labels


def write_labels(path, labels, width, height):
    lines = []
    for c, (x1, y1, x2, y2) in labels:
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        lines.append("%d %.6f %.6f %.6f %.6f" % (
            c, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --- models ---

def boxes_of(result, wanted):
    """[(name, conf, box)] for classes whose name is in `wanted`."""
    out = []
    if result.boxes is None:
        return out
    for c, conf, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        name = result.names[int(c)]
        if name in wanted:
            out.append((name, float(conf), [float(v) for v in box]))
    return out


class Prelabeller:
    def __init__(self, direction_weights, stock_weights="yolo26n.pt"):
        from ultralytics import YOLO
        self.stock = YOLO(stock_weights)
        self.direction = YOLO(direction_weights)
        names = [self.direction.names[k] for k in sorted(self.direction.names)]
        if names[:2] != TWIST_CLASSES[:2]:
            raise ValueError(f"{direction_weights} classes {names} do not start with {TWIST_CLASSES[:2]}")

    def direction_from_crop(self, image, box):
        x1, y1, x2, y2 = box
        pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
        crop = image.crop((max(0, x1 - pw), max(0, y1 - ph),
                           min(image.width, x2 + pw), min(image.height, y2 + ph)))
        dets = boxes_of(self.direction.predict(crop, conf=0.05, verbose=False)[0], TWIST_CLASSES[:2])
        if not dets:
            return None
        name, conf, _ = max(dets, key=lambda d: d[1])
        return TWIST_CLASSES.index(name), conf

    def propose(self, image):
        """Return [(class_id, box, source)] for every person and bicycle found.

        source is "match", "crop" or "direction_unknown" for people, "stock" for bikes.
        """
        stock = boxes_of(self.stock.predict(image, conf=DET_CONF, verbose=False)[0], {"person", "bicycle"})
        direction_dets = [(TWIST_CLASSES.index(n), c, b) for n, c, b in boxes_of(
            self.direction.predict(image, conf=0.01, verbose=False)[0], TWIST_CLASSES[:2])]

        proposals = []
        for name, _, box in stock:
            if name == "bicycle":
                proposals.append((BICYCLE, box, "stock"))
                continue
            found = match_direction(box, direction_dets)
            source = "match"
            if found is None:
                found = self.direction_from_crop(image, box)
                source = "crop"
            if found is None:
                proposals.append((PERSON_LEFT, box, "direction_unknown"))
            else:
                proposals.append((found[0], box, source))
        return proposals


def kind(class_id):
    """People are compared with people, bikes with bikes: a rider overlaps their bike."""
    return "bicycle" if class_id == BICYCLE else "person"


def merge(existing, proposals):
    """Keep every existing label; add proposals that don't duplicate one of the same kind.

    Returns (labels, added) where added is the list of proposals kept.
    """
    labels = list(existing)
    added = []
    for class_id, box, source in proposals:
        same_kind = [b for c, b in labels if kind(c) == kind(class_id)]
        if is_duplicate(box, same_kind):
            continue
        labels.append((class_id, box))
        added.append((class_id, box, source))
    return labels, added


def find_splits(input_dir):
    """[(split_name, images_dir, labels_dir)] for a dataset or a plain image folder."""
    input_dir = Path(input_dir)
    splits = [(s, input_dir / s / "images", input_dir / s / "labels")
              for s in SPLITS if (input_dir / s / "images").is_dir()]
    if splits:
        return splits
    return [("images", input_dir, input_dir.parent / "labels")]


def run(input_dir, out_dir, labeller):
    out_dir = Path(out_dir)
    rows = []
    totals = Counter()
    for split, images_dir, labels_dir in find_splits(input_dir):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            existing = read_labels(labels_dir / (image_path.stem + ".txt"), image.width, image.height)
            labels, added = merge(existing, labeller.propose(image))

            shutil.copy2(image_path, out_dir / split / "images" / image_path.name)
            write_labels(out_dir / split / "labels" / (image_path.stem + ".txt"),
                         labels, image.width, image.height)

            sources = Counter(s for _, _, s in added)
            row = {
                "image": image_path.name, "split": split,
                "existing": len(existing),
                "added_person_left": sum(1 for c, _, _ in added if c == PERSON_LEFT),
                "added_person_right": sum(1 for c, _, _ in added if c == PERSON_RIGHT),
                "added_bicycle": sum(1 for c, _, _ in added if c == BICYCLE),
                "direction_from_crop": sources["crop"],
                "direction_unknown": sources["direction_unknown"],
            }
            rows.append(row)
            totals.update({k: v for k, v in row.items() if isinstance(v, int)})

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"nc": len(TWIST_CLASSES), "names": list(TWIST_CLASSES)}, f, sort_keys=False)

    rows.sort(key=lambda r: -(r["added_person_left"] + r["added_person_right"] + r["added_bicycle"]))
    with open(out_dir / "prelabel_review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="dataset folder (train/valid/test) or a folder of images")
    parser.add_argument("--out", required=True, help="output folder (keep it outside the repo)")
    parser.add_argument("--direction-weights", required=True, help="baseline best.pt (person_left/person_right)")
    parser.add_argument("--stock-weights", default="yolo26n.pt", help="COCO model for boxes (default: %(default)s)")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    if (Path(__file__).resolve().parents[2] / ".git").exists() and Path(__file__).resolve().parents[2] in out.parents:
        parser.error("--out is inside the repo; photos must never be committed. Choose a folder outside it.")
    if Path(args.input).resolve() == out:
        parser.error("--out must be a different folder from the input")

    rows, totals = run(args.input, out, Prelabeller(args.direction_weights, args.stock_weights))
    print(f"images: {len(rows)}  existing labels kept: {totals['existing']}")
    print(f"added: person_left {totals['added_person_left']}, person_right {totals['added_person_right']}, "
          f"bicycle {totals['added_bicycle']}")
    print(f"direction from crop: {totals['direction_from_crop']}, "
          f"unknown (written as person_left, check these): {totals['direction_unknown']}")
    print(f"wrote {out} (data.yaml, prelabel_review.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
