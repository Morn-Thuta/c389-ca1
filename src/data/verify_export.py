"""Sanity-check a Roboflow YOLO export before training (PRD §5.4.1).

Checks:
  - data.yaml class order matches config.CLASS_NAMES (2-class baseline or 4-class twist)
  - image and box counts per split and per class
  - empty label files, images without a label file, labels without an image
  - boxes shorter than MIN_BOX_HEIGHT_FRAC of the image height
  - label IDs outside the class range, malformed label lines

Errors (wrong class order, bad IDs, malformed lines) exit 1. Everything else is a
warning, which also exits 1 with --strict.

Usage:
    python -m src.data.verify_export /content/drive/MyDrive/C389/runs/L3/dataset
    python -m src.data.verify_export DATASET_DIR --expect 4 --json reports/verify_export.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from src.config import CLASS_NAMES, IMAGE_EXTS, MIN_BOX_HEIGHT_FRAC, SPLITS


def load_class_names(dataset_dir):
    """Return the class names from data.yaml as a list (handles list or {id: name} form)."""
    with open(Path(dataset_dir) / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def check_class_order(names, expect=None):
    """Return an error string, or None if the order is valid."""
    if expect is not None and len(names) != expect:
        return f"expected {expect} classes, data.yaml has {len(names)}: {names}"
    if len(names) not in (2, len(CLASS_NAMES)) or names != CLASS_NAMES[: len(names)]:
        return f"class order {names} does not match {CLASS_NAMES[:max(len(names), 2)]}"
    return None


def check_split(split_dir, num_classes):
    """Scan one split. Returns a dict of counts, warnings and errors."""
    images_dir = Path(split_dir) / "images"
    labels_dir = Path(split_dir) / "labels"
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS} if images_dir.is_dir() else {}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.is_dir() else {}

    result = {
        "images": len(images),
        "boxes": Counter(),
        "empty_labels": [],
        "missing_labels": sorted(set(images) - set(labels)),
        "orphan_labels": sorted(set(labels) - set(images)),
        "small_boxes": [],
        "bad_ids": [],
        "malformed": [],
    }

    for stem in sorted(set(images) & set(labels)):
        lines = [ln.strip() for ln in labels[stem].read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            result["empty_labels"].append(stem)
            continue
        for lineno, line in enumerate(lines, 1):
            where = f"{stem}.txt:{lineno}"
            parts = line.split()
            if len(parts) != 5:
                result["malformed"].append(f"{where} ({len(parts)} values, expected 5)")
                continue
            try:
                cls_id = int(parts[0])
                h = float(parts[4])
            except ValueError:
                result["malformed"].append(f"{where} (not numeric)")
                continue
            if not 0 <= cls_id < num_classes:
                result["bad_ids"].append(f"{where} (class {cls_id})")
                continue
            result["boxes"][CLASS_NAMES[cls_id]] += 1
            if h < MIN_BOX_HEIGHT_FRAC:
                result["small_boxes"].append(f"{where} ({CLASS_NAMES[cls_id]}, h={h:.3f})")

    return result


def verify(dataset_dir, expect=None):
    """Run every check. Returns a report dict with 'errors', 'warnings' and per-split results."""
    dataset_dir = Path(dataset_dir)
    report = {"dataset": str(dataset_dir), "errors": [], "warnings": [], "splits": {}}

    if not (dataset_dir / "data.yaml").is_file():
        report["errors"].append(f"no data.yaml in {dataset_dir}")
        return report

    names = load_class_names(dataset_dir)
    report["class_names"] = names
    order_error = check_class_order(names, expect)
    if order_error:
        report["errors"].append(order_error)

    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            report["warnings"].append(f"{split}: split folder missing")
            continue
        res = check_split(split_dir, len(names))
        report["splits"][split] = res
        for key in ("bad_ids", "malformed"):
            if res[key]:
                report["errors"].append(f"{split}: {len(res[key])} {key.replace('_', ' ')}")
        for key in ("empty_labels", "missing_labels", "orphan_labels", "small_boxes"):
            if res[key]:
                report["warnings"].append(f"{split}: {len(res[key])} {key.replace('_', ' ')}")

    return report


def print_report(report, limit=10):
    names = report.get("class_names", [])
    print(f"Dataset: {report['dataset']}")
    if names:
        print(f"Classes: {names}")
    if report["splits"]:
        header = f"{'split':<8}{'images':>8}" + "".join(f"{n:>14}" for n in names) + f"{'total':>8}"
        print("\n" + header)
        print("-" * len(header))
        totals = Counter()
        total_images = 0
        for split, res in report["splits"].items():
            boxes = res["boxes"]
            totals.update(boxes)
            total_images += res["images"]
            row = f"{split:<8}{res['images']:>8}" + "".join(f"{boxes[n]:>14}" for n in names)
            print(row + f"{sum(boxes.values()):>8}")
        print("-" * len(header))
        print(f"{'all':<8}{total_images:>8}" + "".join(f"{totals[n]:>14}" for n in names) + f"{sum(totals.values()):>8}")

        for split, res in report["splits"].items():
            for key in ("bad_ids", "malformed", "empty_labels", "missing_labels", "orphan_labels", "small_boxes"):
                items = res[key]
                if items:
                    print(f"\n[{split}] {key} ({len(items)}):")
                    for item in items[:limit]:
                        print(f"  {item}")
                    if len(items) > limit:
                        print(f"  ... and {len(items) - limit} more")

    print()
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for e in report["errors"]:
        print(f"ERROR: {e}")
    if not report["errors"] and not report["warnings"]:
        print("OK: no problems found")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", help="folder containing data.yaml and train/valid/test")
    parser.add_argument("--expect", type=int, choices=(2, 4), help="require exactly this many classes")
    parser.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    parser.add_argument("--json", help="also write the full report to this JSON file")
    parser.add_argument("--limit", type=int, default=10, help="max items listed per problem (default 10)")
    args = parser.parse_args(argv)

    report = verify(args.dataset_dir, args.expect)
    print_report(report, args.limit)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    failed = bool(report["errors"]) or (args.strict and bool(report["warnings"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
