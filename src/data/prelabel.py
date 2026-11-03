"""Pre-draw labels for every person and wheeled crosser, for humans to correct in Roboflow.

  - Boxes come from a stock COCO model (yolo26n.pt): it finds people and bikes well,
    but cannot tell left from right.
  - Direction comes from our baseline (best.pt): each stock person box is matched
    to the baseline box that overlaps it most. If none overlaps enough, the person
    is cropped and the baseline is run again on the crop. If that also fails, the
    box is written as *_left and flagged "direction_unknown".
  - A person with a bike, riding or walking it (src/linking.py), becomes one
    wheeled_* box around both, in their direction. Bikes with nobody at them are
    not labelled.
  - Existing (human) labels are kept as they are. A proposed box that overlaps an
    existing label is dropped, so only the missing people and riders are added.

Output is a YOLO dataset in our class order (person_left, person_right,
wheeled_left, wheeled_right) that can be dropped into Roboflow's upload page, plus
prelabel_review.csv listing what was added to each image so reviewers can start
with the busiest ones.

Existing labels are read by class *name* from the input's data.yaml, so a Roboflow
export in alphabetical order is converted to our order first. An export that still
has a separate `bicycle` class is refused: run src/data/to_wheeled.py on it first.

--split-log applies the v2 cleaning to an original export: images listed in
split_log.csv go to their `new_split`, and images not listed (the removed
duplicates) are skipped. File names are matched without Roboflow's ".rf.<hash>"
suffix and extension tokens, so "9_jpeg" matches "9_jpeg_jpg".

**Every image must still be checked by a person.** Use the original photos, not a
masked export: blurred people cannot be found.

Usage:
    python -m src.data.prelabel DATASET_DIR --out OUT_DIR --direction-weights best.pt
    python -m src.data.prelabel IMAGES_DIR  --out OUT_DIR --direction-weights best.pt
    python -m src.data.prelabel V1_EXPORT --split-log v2_cleaned/split_log.csv --out OUT_DIR --direction-weights best.pt
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from src.config import CLASS_NAMES, IMAGE_EXTS, SPLITS
from src.data.remap_classes import class_mapping, load_names
from src.linking import link_people_to_bikes, union_box

PERSON_LEFT, PERSON_RIGHT, WHEELED_LEFT, WHEELED_RIGHT = 0, 1, 2, 3
WHEELED_OF = {PERSON_LEFT: WHEELED_LEFT, PERSON_RIGHT: WHEELED_RIGHT}

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

def read_labels(path, width, height, mapping=None):
    """Return [(class_id, [x1, y1, x2, y2])] in pixels; [] if the file is missing.

    mapping converts the file's class ids to ours ({their_id: our_id}).
    """
    labels = []
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        if mapping is not None:
            c = mapping[c]
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
    def __init__(self, direction_weights, stock_weights="yolo26n.pt", stock_imgsz=640):
        from ultralytics import YOLO
        self.stock = YOLO(stock_weights)
        self.stock_imgsz = stock_imgsz  # larger finds small, far-away people and bikes
        self.direction = YOLO(direction_weights)
        names = [self.direction.names[k] for k in sorted(self.direction.names)]
        if names[:2] != CLASS_NAMES[:2]:
            raise ValueError(f"{direction_weights} classes {names} do not start with {CLASS_NAMES[:2]}")

    def direction_from_crop(self, image, box):
        x1, y1, x2, y2 = box
        pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
        crop = image.crop((max(0, x1 - pw), max(0, y1 - ph),
                           min(image.width, x2 + pw), min(image.height, y2 + ph)))
        dets = boxes_of(self.direction.predict(crop, conf=0.05, verbose=False)[0], CLASS_NAMES[:2])
        if not dets:
            return None
        name, conf, _ = max(dets, key=lambda d: d[1])
        return CLASS_NAMES.index(name), conf

    def propose(self, image):
        """Return [(class_id, box, source)] for every person and rider found.

        source is "match", "crop" or "direction_unknown": how the direction was found.
        """
        stock = boxes_of(self.stock.predict(image, conf=DET_CONF, imgsz=self.stock_imgsz, verbose=False)[0],
                         {"person", "bicycle"})
        direction_dets = [(CLASS_NAMES.index(n), c, b) for n, c, b in boxes_of(
            self.direction.predict(image, conf=0.01, verbose=False)[0], CLASS_NAMES[:2])]

        people, bikes = [], []
        for name, _, box in stock:
            if name == "bicycle":
                bikes.append(box)
                continue
            found = match_direction(box, direction_dets)
            source = "match"
            if found is None:
                found = self.direction_from_crop(image, box)
                source = "crop"
            if found is None:
                found, source = (PERSON_LEFT, None), "direction_unknown"
            people.append((found[0], box, source))

        links = link_people_to_bikes([b for _, b, _ in people], bikes)
        proposals = []
        for i, (class_id, box, source) in enumerate(people):
            if i in links:
                proposals.append((WHEELED_OF[class_id], union_box(box, bikes[links[i][0]]), source))
            else:
                proposals.append((class_id, box, source))
        return proposals


def merge(existing, proposals):
    """Keep every existing label; add proposals that don't duplicate one.

    A rider sits inside their wheeled box, so a person proposal inside an existing
    wheeled label counts as a duplicate too. Returns (labels, added) where added
    is the list of proposals kept.
    """
    labels = list(existing)
    added = []
    for class_id, box, source in proposals:
        if is_duplicate(box, [b for _, b in labels]):
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
    # a plain folder of photos: output goes straight into OUT/images and OUT/labels
    return [("", input_dir, input_dir.parent / "labels")]


EXT_TOKENS = {"jpg", "jpeg", "png", "bmp", "webp", "heic"}


def image_key(filename):
    """Name without Roboflow's ".rf.<hash>" suffix and any extension it appended.

    Each Roboflow re-upload appends "_jpg": "9_jpeg.rf.ab12.jpg" and
    "9_jpeg_jpg.rf.cd34.jpg" both give "9_jpeg". The original extension token is
    kept, so "2_jpg" and "2_jpeg" (two different photos) stay apart.
    """
    stem = Path(filename).name.split(".rf.")[0]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    parts = stem.split("_")
    while len(parts) > 2 and parts[-1].lower() in EXT_TOKENS and parts[-2].lower() in EXT_TOKENS:
        parts.pop()
    return "_".join(parts)


def read_split_log(path):
    """{image_key: new_split} from the v2 cleaning's split_log.csv."""
    with open(path, newline="", encoding="utf-8") as f:
        return {image_key(r["file"]): r["new_split"] for r in csv.DictReader(f)}


def input_mapping(input_dir):
    """Class-id mapping for the input's labels, or None if it has no data.yaml."""
    if not (Path(input_dir) / "data.yaml").is_file():
        return None
    return class_mapping(load_names(input_dir))


def run(input_dir, out_dir, labeller, split_log=None):
    """Pre-label every image. split_log ({key: split}) re-splits and drops unlisted images.

    Returns (rows, totals, skipped) where skipped lists images left out by split_log.
    """
    out_dir = Path(out_dir)
    mapping = input_mapping(input_dir)
    rows, skipped = [], []
    totals = Counter()
    for split, images_dir, labels_dir in find_splits(input_dir):
        for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            out_split = split
            if split_log is not None:
                out_split = split_log.get(image_key(image_path.name))
                if out_split is None:
                    skipped.append(image_path.name)
                    continue
            (out_dir / out_split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / out_split / "labels").mkdir(parents=True, exist_ok=True)

            with Image.open(image_path) as img:
                # Roboflow auto-orients phone photos, so measure boxes on the upright image
                image = ImageOps.exif_transpose(img).convert("RGB")
            existing = read_labels(labels_dir / (image_path.stem + ".txt"),
                                   image.width, image.height, mapping)
            labels, added = merge(existing, labeller.propose(image))

            shutil.copy2(image_path, out_dir / out_split / "images" / image_path.name)
            write_labels(out_dir / out_split / "labels" / (image_path.stem + ".txt"),
                         labels, image.width, image.height)

            sources = Counter(s for _, _, s in added)
            row = {
                "image": image_path.name, "split": out_split or "-",
                "existing": len(existing),
                "added_person_left": sum(1 for c, _, _ in added if c == PERSON_LEFT),
                "added_person_right": sum(1 for c, _, _ in added if c == PERSON_RIGHT),
                "added_wheeled_left": sum(1 for c, _, _ in added if c == WHEELED_LEFT),
                "added_wheeled_right": sum(1 for c, _, _ in added if c == WHEELED_RIGHT),
                "direction_from_crop": sources["crop"],
                "direction_unknown": sources["direction_unknown"],
            }
            rows.append(row)
            totals.update({k: v for k, v in row.items() if isinstance(v, int)})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "train/images", "val": "valid/images", "test": "test/images",
                        "nc": len(CLASS_NAMES), "names": list(CLASS_NAMES)}, f, sort_keys=False)

    added_keys = ["added_person_left", "added_person_right", "added_wheeled_left", "added_wheeled_right"]
    rows.sort(key=lambda r: -sum(r[k] for k in added_keys))
    with open(out_dir / "prelabel_review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, totals, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="dataset folder (train/valid/test) or a folder of images")
    parser.add_argument("--out", required=True, help="output folder (keep it outside the repo)")
    parser.add_argument("--direction-weights", required=True, help="baseline best.pt (person_left/person_right)")
    parser.add_argument("--stock-weights", default="yolo26n.pt", help="COCO model for boxes (default: %(default)s)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="stock model input size; 1280 finds more small boxes in large photos (default: %(default)s)")
    parser.add_argument("--split-log", help="split_log.csv from the v2 cleaning: re-split, and drop unlisted images")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    if (Path(__file__).resolve().parents[2] / ".git").exists() and Path(__file__).resolve().parents[2] in out.parents:
        parser.error("--out is inside the repo; photos must never be committed. Choose a folder outside it.")
    if Path(args.input).resolve() == out:
        parser.error("--out must be a different folder from the input")

    try:
        mapping = input_mapping(args.input)
    except ValueError as e:
        parser.error(str(e))
    if mapping is not None and any(k != v for k, v in mapping.items()):
        names = load_names(args.input)
        print("input classes remapped: " + ", ".join(f"{names[i]} {i} -> {j}" for i, j in mapping.items()))
    split_log = read_split_log(args.split_log) if args.split_log else None

    rows, totals, skipped = run(args.input, out, Prelabeller(args.direction_weights, args.stock_weights, args.imgsz), split_log)
    if skipped:
        print(f"skipped {len(skipped)} images not in the split log: {', '.join(skipped)}")
    by_split = Counter(r["split"] for r in rows)
    print(f"images: {len(rows)} ({', '.join(f'{s} {n}' for s, n in sorted(by_split.items()))})  "
          f"existing labels kept: {totals['existing']}")
    print(f"added: person_left {totals['added_person_left']}, person_right {totals['added_person_right']}, "
          f"wheeled_left {totals['added_wheeled_left']}, wheeled_right {totals['added_wheeled_right']}")
    print(f"direction from crop: {totals['direction_from_crop']}, "
          f"unknown (written as *_left, check these): {totals['direction_unknown']}")
    print(f"wrote {out} (data.yaml, prelabel_review.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
