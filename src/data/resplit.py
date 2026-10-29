"""Put a Roboflow export back into session-based splits (PRD §5).

Generating a new version in Roboflow can re-split images at random, which mixes
burst shots from one session across train, valid and test. This copies the
export into a new folder where each image goes to:
  1. its split in the v2 split_log.csv (images the team already split), else
  2. the split given for its session with --session DATE=SPLIT.
A session is the capture date in names like "photo_12_2026-10-29_13-48-00.jpg".
Any image matched by neither is reported and nothing is written, so no image is
ever split by guesswork.

Usage (after remap_classes, before verify_export and dedupe_check):
    python -m src.data.resplit FIXED_DIR --out RESPLIT_DIR \\
        --split-log path/to/v2_cleaned/split_log.csv --session 2026-10-29=train
"""

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

from src.config import IMAGE_EXTS, SPLITS
from src.data.prelabel import image_key, read_split_log
from src.data.remap_classes import load_names

SESSION_RE = re.compile(r"photo_\d+_(\d{4}-\d{2}-\d{2})_")


def session_of(key):
    """Capture date of a phone-named photo, or None for other names."""
    m = SESSION_RE.match(key)
    return m.group(1) if m else None


def parse_session(text):
    """'2026-10-29=train' -> ('2026-10-29', 'train')."""
    date, _, split = text.partition("=")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or split not in SPLITS:
        raise argparse.ArgumentTypeError(f"expected DATE=SPLIT with SPLIT in {SPLITS}, got {text!r}")
    return date, split


def target_split(key, split_log, session_splits):
    """(split, source) for one image, or (None, None) if nothing matches."""
    if key in split_log:
        return split_log[key], "split_log"
    session = session_of(key)
    if session in session_splits:
        return session_splits[session], f"session {session}"
    return None, None


def plan(src_dir, split_log, session_splits):
    """Rows {file, key, old_split, new_split, source} for every image, plus unmatched files."""
    rows, unmatched = [], []
    for split in SPLITS:
        images = Path(src_dir) / split / "images"
        if not images.is_dir():
            continue
        for path in sorted(images.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            key = image_key(path.name)
            new_split, source = target_split(key, split_log, session_splits)
            if new_split is None:
                unmatched.append(f"{split}/{path.name}")
                continue
            rows.append({"file": path.name, "key": key, "old_split": split,
                         "new_split": new_split, "source": source})
    return rows, unmatched


def copy_dataset(src_dir, out_dir, rows):
    """Copy each image and its label file into its new split; write data.yaml and the log."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    for row in rows:
        src_img = src_dir / row["old_split"] / "images" / row["file"]
        src_lbl = src_dir / row["old_split"] / "labels" / (src_img.stem + ".txt")
        dst = out_dir / row["new_split"]
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst / "images" / row["file"])
        if src_lbl.is_file():
            shutil.copy2(src_lbl, dst / "labels" / src_lbl.name)

    names = load_names(src_dir)
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "train/images", "val": "valid/images", "test": "test/images",
                        "nc": len(names), "names": names}, f, sort_keys=False)
    with open(out_dir / "resplit_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "key", "old_split", "new_split", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", help="YOLO export with train/valid/test (run remap_classes first)")
    parser.add_argument("--out", required=True, help="new folder for the re-split copy (outside the repo)")
    parser.add_argument("--split-log", help="split_log.csv from the v2 cleaning")
    parser.add_argument("--session", action="append", type=parse_session, default=[], metavar="DATE=SPLIT",
                        help="split for every photo of a session not in the split log (repeatable)")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parents[2]
    if (repo / ".git").exists() and repo in out.parents:
        parser.error("--out is inside the repo; photos must never be committed. Choose a folder outside it.")
    if Path(args.export_dir).resolve() == out:
        parser.error("--out must be a different folder from the export")
    if out.exists() and any(out.iterdir()):
        parser.error(f"--out {out} is not empty; choose a new folder")

    split_log = read_split_log(args.split_log) if args.split_log else {}
    rows, unmatched = plan(args.export_dir, split_log, dict(args.session))
    if unmatched:
        sessions = sorted({session_of(image_key(f.split("/", 1)[1])) or "no session in name" for f in unmatched})
        print(f"{len(unmatched)} images have no split (sessions: {', '.join(sessions)}):")
        for f in unmatched:
            print(f"  {f}")
        print("Add --session DATE=SPLIT for each session, or add the images to the split log. Nothing written.")
        return 1

    copy_dataset(args.export_dir, out, rows)
    moved = sum(1 for r in rows if r["old_split"] != r["new_split"])
    by_split = Counter(r["new_split"] for r in rows)
    print(f"images: {len(rows)} ({', '.join(f'{s} {by_split[s]}' for s in SPLITS if by_split[s])}), "
          f"moved from Roboflow's split: {moved}")
    missing = sorted(set(split_log) - {r["key"] for r in rows})
    if missing:
        print(f"in the split log but not in this export ({len(missing)}): {', '.join(missing)}")
    print(f"wrote {out} (data.yaml, resplit_log.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
