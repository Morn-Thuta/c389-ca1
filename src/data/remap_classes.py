"""Put a YOLO export's class ids into our fixed order (PRD §4).

Roboflow lists classes alphabetically, so a 3-class export comes out as
['bicycle', 'person_left', 'person_right'] with bicycle = 0. Our notebooks and
models need person_left = 0, person_right = 1, bicycle = 2. This rewrites every
label file by class *name*, and writes a data.yaml in our order.

Usage (run after every Roboflow export, before verify_export and training):
    python -m src.data.remap_classes EXPORT_DIR --out FIXED_DIR
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from src.config import CLASS_NAMES, SPLITS


def load_names(dataset_dir):
    """Class names from data.yaml as a list (handles list or {id: name})."""
    with open(Path(dataset_dir) / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def class_mapping(names):
    """{their_id: our_id} for an export whose classes are `names`.

    Raises ValueError for a class we don't have, so a new class (for example a
    team-added person_other) is never silently mislabelled.
    """
    unknown = [n for n in names if n not in CLASS_NAMES]
    if unknown:
        raise ValueError(f"classes {unknown} are not in {CLASS_NAMES}; add them to src/config.py first")
    return {i: CLASS_NAMES.index(n) for i, n in enumerate(names)}


def target_names(names):
    """Our class list, long enough to cover every class the export uses."""
    return CLASS_NAMES[: max(CLASS_NAMES.index(n) for n in names) + 1]


def remap_line(line, mapping):
    parts = line.split()
    if not parts:
        return None
    parts[0] = str(mapping[int(parts[0])])
    return " ".join(parts)


def remap_file(src, dst, mapping):
    lines = [remap_line(ln, mapping) for ln in Path(src).read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    Path(dst).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def remap_dataset(src_dir, out_dir):
    """Copy src_dir to out_dir with labels in our class order. Returns (mapping, n_files)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    names = load_names(src_dir)
    mapping = class_mapping(names)
    count = 0
    for split in SPLITS:
        if not (src_dir / split).is_dir():
            continue
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        if (src_dir / split / "images").is_dir():
            shutil.copytree(src_dir / split / "images", out_dir / split / "images", dirs_exist_ok=True)
        for label in (src_dir / split / "labels").glob("*.txt"):
            remap_file(label, out_dir / split / "labels" / label.name, mapping)
            count += 1
    new_names = target_names(names)
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "train/images", "val": "valid/images", "test": "test/images",
                        "nc": len(new_names), "names": new_names}, f, sort_keys=False)
    return mapping, count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", help="Roboflow YOLO export (folder with data.yaml)")
    parser.add_argument("--out", required=True, help="new folder for the remapped copy")
    args = parser.parse_args(argv)
    if Path(args.out).resolve() == Path(args.export_dir).resolve():
        parser.error("--out must be a different folder")

    names = load_names(args.export_dir)
    try:
        mapping, count = remap_dataset(args.export_dir, args.out)
    except ValueError as e:
        parser.error(str(e))
    print("mapping: " + ", ".join(f"{names[i]} {i} -> {j}" for i, j in mapping.items()))
    print(f"rewrote {count} label files into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
