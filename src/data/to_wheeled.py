"""Convert an export with a separate `bicycle` class to the Twist 2 classes.

Older versions labelled a cyclist as a person box plus a bicycle box. The twist
classes (PRD §4) put a rider and their bike in one box with a direction:
  - a person riding a bike (src/linking.py)  -> one wheeled_left / wheeled_right
    box around both, in the person's direction
  - a bike nobody rides (parked, pushed)      -> removed
  - everyone else                             -> stays person_left / person_right
Classes are read by *name*, so Roboflow's alphabetical order is handled. The
output is in our class order, ready for resplit and verify_export.

to_wheeled_review.csv lists every image where a rider was merged or a bike was
removed, so the team can check those first in Roboflow. **The linking rule was
tuned for detections, so check every converted image.**

Usage (instead of remap_classes, for an export that still has `bicycle`):
    python -m src.data.to_wheeled EXPORT_DIR --out WHEELED_DIR
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

from src.config import CLASS_NAMES, LEGACY_BICYCLE, PERSON_CLASSES, SPLITS, WHEELED_CLASSES
from src.data.remap_classes import load_names
from src.linking import link_people_to_bikes, union_box


def read_boxes(path, names):
    """[(class_name, [x1, y1, x2, y2])] in normalised corners."""
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((names[int(parts[0])], [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return boxes


def write_boxes(path, boxes):
    lines = []
    for name, (x1, y1, x2, y2) in boxes:
        lines.append("%d %.6f %.6f %.6f %.6f" % (CLASS_NAMES.index(name), (x1 + x2) / 2, (y1 + y2) / 2,
                                                  x2 - x1, y2 - y1))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert(boxes):
    """One image's boxes -> (new boxes, riders merged, bikes removed)."""
    people = [(n, b) for n, b in boxes if n in PERSON_CLASSES]
    bikes = [b for n, b in boxes if n == LEGACY_BICYCLE]
    kept = [(n, b) for n, b in boxes if n in WHEELED_CLASSES]
    links = link_people_to_bikes([b for _, b in people], bikes)
    for i, (name, box) in enumerate(people):
        if i in links:
            kept.append(("wheeled_" + name.rsplit("_", 1)[1], union_box(box, bikes[links[i][0]])))
        else:
            kept.append((name, box))
    return kept, len(links), len(bikes) - len(links)


def convert_dataset(src_dir, out_dir):
    """Convert every split. Returns (review rows, totals Counter)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    names = load_names(src_dir)
    unknown = [n for n in names if n not in CLASS_NAMES and n != LEGACY_BICYCLE]
    if unknown:
        raise ValueError(f"classes {unknown} are not in {CLASS_NAMES + [LEGACY_BICYCLE]}")

    rows, totals = [], Counter()
    for split in SPLITS:
        if not (src_dir / split / "images").is_dir():
            continue
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir / split / "images", out_dir / split / "images", dirs_exist_ok=True)
        for label in sorted((src_dir / split / "labels").glob("*.txt")):
            boxes, merged, removed = convert(read_boxes(label, names))
            write_boxes(out_dir / split / "labels" / label.name, boxes)
            totals.update(Counter(n for n, _ in boxes))
            totals.update({"riders_merged": merged, "bikes_removed": removed, "images": 1})
            if merged or removed:
                rows.append({"image": label.stem, "split": split,
                             "riders_merged": merged, "bikes_removed": removed})

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "train/images", "val": "valid/images", "test": "test/images",
                        "nc": len(CLASS_NAMES), "names": list(CLASS_NAMES)}, f, sort_keys=False)
    with open(out_dir / "to_wheeled_review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "riders_merged", "bikes_removed"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", help="YOLO export with person_left, person_right and bicycle")
    parser.add_argument("--out", required=True, help="new folder for the converted copy (outside the repo)")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parents[2]
    if (repo / ".git").exists() and repo in out.parents:
        parser.error("--out is inside the repo; photos must never be committed. Choose a folder outside it.")
    if Path(args.export_dir).resolve() == out:
        parser.error("--out must be a different folder from the export")
    try:
        rows, totals = convert_dataset(args.export_dir, out)
    except ValueError as e:
        parser.error(str(e))

    print(f"images: {totals['images']}  riders merged into wheeled boxes: {totals['riders_merged']}  "
          f"unridden bikes removed: {totals['bikes_removed']}")
    print("boxes: " + ", ".join(f"{n} {totals[n]}" for n in CLASS_NAMES))
    print(f"wrote {out} (data.yaml, to_wheeled_review.csv: {len(rows)} images to check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
