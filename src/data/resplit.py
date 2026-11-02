"""Re-split a Roboflow export without leaking burst shots across splits (PRD §5).

Generating a new version in Roboflow can re-split images at random, which mixes
near-identical burst shots across train, valid and test. This copies the export
into a new folder, choosing each image's split in one of two ways.

--ratio 70/20/10 (the team rule):
  Images within DEDUPE_MAX_HAMMING of each other (by perceptual hash, chained)
  form one burst group, and a group always goes to one split. Within each
  session, groups that contain a bicycle are shared out first so bikes land in
  every split at the ratio, then the rest are shared out by image count. The
  same --seed always gives the same split.

--split-log / --session (fixed assignments):
  1. its split in the v2 split_log.csv (images the team already split), else
  2. the split given for its session with --session DATE=SPLIT.
  Any image matched by neither is reported and nothing is written.

A session is the capture date in names like "photo_12_2026-10-29_13-48-00.jpg";
other names share one "undated" session.

Usage (after remap_classes, before verify_export and dedupe_check):
    python -m src.data.resplit FIXED_DIR --out RESPLIT_DIR --ratio 70/20/10
"""

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import imagehash
import yaml
from PIL import Image

from src.config import DEDUPE_MAX_HAMMING, IMAGE_EXTS, SPLITS
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


def parse_ratio(text):
    """'70/20/10' -> [0.7, 0.2, 0.1] for train/valid/test."""
    try:
        parts = [float(x) for x in text.split("/")]
    except ValueError:
        parts = []
    if len(parts) != len(SPLITS) or any(x < 0 for x in parts) or sum(parts) <= 0:
        raise argparse.ArgumentTypeError(f"expected TRAIN/VALID/TEST like 70/20/10, got {text!r}")
    return [x / sum(parts) for x in parts]


def list_images(src_dir):
    """[(split, path)] for every image in the export."""
    found = []
    for split in SPLITS:
        images = Path(src_dir) / split / "images"
        if images.is_dir():
            found += [(split, p) for p in sorted(images.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return found


def burst_groups(hashes, max_distance=DEDUPE_MAX_HAMMING):
    """Group indexes whose hashes are within max_distance, chaining (union-find)."""
    parent = list(range(len(hashes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(hashes)), 2):
        if hashes[i] - hashes[j] <= max_distance:
            parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i in range(len(hashes)):
        groups[root(i)].append(i)
    return sorted(groups.values())


def share_out(groups, weight, ratios, totals):
    """Give each group to the split furthest below its share of `weight`.

    groups are lists of image indexes, weight(group) is what is balanced, and
    totals ({split: weight so far}) is updated in place. Returns {index: split}.
    """
    assigned = {}
    for group in sorted(groups, key=weight, reverse=True):
        w = weight(group)
        after = sum(totals.values()) + w
        split = max(SPLITS, key=lambda s: ratios[SPLITS.index(s)] * after - totals[s])
        totals[split] += w
        for i in group:
            assigned[i] = split
    return assigned


def bikes_in(label_path, bicycle_id):
    if bicycle_id is None or not label_path.is_file():
        return 0
    return sum(1 for ln in label_path.read_text(encoding="utf-8").splitlines()
               if ln.split() and int(ln.split()[0]) == bicycle_id)


def ratio_plan(src_dir, ratios, seed=0, max_distance=DEDUPE_MAX_HAMMING):
    """Rows like plan(), with burst groups shared out per session at `ratios`."""
    images = list_images(src_dir)
    names = load_names(src_dir)
    bicycle_id = names.index("bicycle") if "bicycle" in names else None
    keys = [image_key(p.name) for _, p in images]
    bikes = [bikes_in(p.parent.parent / "labels" / (p.stem + ".txt"), bicycle_id) for _, p in images]
    hashes = []
    for _, p in images:
        with Image.open(p) as img:
            hashes.append(imagehash.phash(img))
    groups = burst_groups(hashes, max_distance)

    # a group spanning sessions is filed under its first image's session
    by_session = defaultdict(list)
    for group in groups:
        by_session[session_of(keys[group[0]]) or "undated"].append(group)

    rng = random.Random(seed)
    assigned = {}
    for session in sorted(by_session):
        session_groups = by_session[session]
        rng.shuffle(session_groups)  # equal-weight groups are then ordered by seed, not by name
        n_bikes = lambda g: sum(bikes[i] for i in g)
        with_bikes = [g for g in session_groups if n_bikes(g)]
        without = [g for g in session_groups if not n_bikes(g)]
        assigned.update(share_out(with_bikes, n_bikes, ratios, dict.fromkeys(SPLITS, 0)))
        placed = Counter(assigned[i] for g in with_bikes for i in g)
        assigned.update(share_out(without, len, ratios, {s: placed[s] for s in SPLITS}))

    group_of = {i: n for n, g in enumerate(groups) for i in g}
    return [{"file": p.name, "key": keys[i], "old_split": split, "new_split": assigned[i],
             "source": f"group {group_of[i]}"} for i, (split, p) in enumerate(images)]


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
    parser.add_argument("--ratio", type=parse_ratio, metavar="TRAIN/VALID/TEST",
                        help="share burst groups out at this ratio, e.g. 70/20/10")
    parser.add_argument("--seed", type=int, default=0, help="seed for --ratio (default: %(default)s)")
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

    if args.ratio and (args.split_log or args.session):
        parser.error("use either --ratio or --split-log/--session, not both")
    if not args.ratio and not (args.split_log or args.session):
        parser.error("give --ratio, or --split-log and/or --session")

    split_log = read_split_log(args.split_log) if args.split_log else {}
    if args.ratio:
        rows, unmatched = ratio_plan(args.export_dir, args.ratio, args.seed), []
    else:
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
    if args.ratio:
        print(f"burst groups: {len({r['source'] for r in rows})} (each group is in one split)")
    missing = sorted(set(split_log) - {r["key"] for r in rows})
    if missing:
        print(f"in the split log but not in this export ({len(missing)}): {', '.join(missing)}")
    print(f"wrote {out} (data.yaml, resplit_log.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
