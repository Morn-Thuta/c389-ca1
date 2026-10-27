"""Find near-duplicate images that sit in different splits (PRD §5.4.2).

Burst shots of the same scene landing in both train and test leak the answer and
inflate test scores. This hashes every image with a perceptual hash and reports any
cross-split pair within DEDUPE_MAX_HAMMING bits. Exits 1 if any are found.

Usage:
    python -m src.data.dedupe_check DATASET_DIR
    python -m src.data.dedupe_check DATASET_DIR --max-distance 4 --csv reports/dedupe.csv
"""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image

from src.config import DEDUPE_MAX_HAMMING, IMAGE_EXTS, SPLITS


def hash_split(images_dir):
    """Return {filename: phash} for every image in a folder."""
    hashes = {}
    for path in sorted(Path(images_dir).iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        with Image.open(path) as img:
            hashes[path.name] = imagehash.phash(img)
    return hashes


def find_cross_split_duplicates(dataset_dir, max_distance=DEDUPE_MAX_HAMMING):
    """Return (pairs, counts): pairs sorted by distance, counts = images hashed per split."""
    dataset_dir = Path(dataset_dir)
    hashes = {}
    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        if images_dir.is_dir():
            hashes[split] = hash_split(images_dir)

    pairs = []
    for split_a, split_b in combinations(hashes, 2):
        for name_a, hash_a in hashes[split_a].items():
            for name_b, hash_b in hashes[split_b].items():
                distance = int(hash_a - hash_b)
                if distance <= max_distance:
                    pairs.append({
                        "split_a": split_a, "image_a": name_a,
                        "split_b": split_b, "image_b": name_b,
                        "distance": distance,
                    })
    pairs.sort(key=lambda p: (p["distance"], p["split_a"], p["image_a"]))
    counts = {split: len(h) for split, h in hashes.items()}
    return pairs, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", help="folder containing train/valid/test")
    parser.add_argument("--max-distance", type=int, default=DEDUPE_MAX_HAMMING,
                        help=f"max Hamming distance counted as duplicate (default {DEDUPE_MAX_HAMMING})")
    parser.add_argument("--csv", help="also write the pairs to this CSV file")
    args = parser.parse_args(argv)

    pairs, counts = find_cross_split_duplicates(args.dataset_dir, args.max_distance)
    if not counts:
        print(f"ERROR: no train/valid/test image folders found in {args.dataset_dir}")
        return 1

    print("Images hashed: " + ", ".join(f"{s}={n}" for s, n in counts.items()))
    print(f"Max Hamming distance: {args.max_distance}\n")
    for p in pairs:
        print(f"  d={p['distance']:<2} {p['split_a']}/{p['image_a']}  <->  {p['split_b']}/{p['image_b']}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split_a", "image_a", "split_b", "image_b", "distance"])
            writer.writeheader()
            writer.writerows(pairs)

    if pairs:
        print(f"\nFAIL: {len(pairs)} cross-split near-duplicate pair(s). Move them into the same split.")
        return 1
    print("OK: 0 cross-split near-duplicates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
