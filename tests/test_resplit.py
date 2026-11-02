"""Tests for src/data/resplit.py: session-based re-splitting of an export."""

import csv

import pytest
import yaml
from PIL import Image

from src.data import resplit


def make_export(root, files):
    """files: {split: [image names]}; each image gets a one-box label."""
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 2, "names": ["person_left", "person_right"]}))
    for split, names in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name in names:
            Image.new("RGB", (20, 20)).save(root / split / "images" / name)
            stem = name.rsplit(".", 1)[0]
            (root / split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.4\n")


def test_session_of():
    assert resplit.session_of("photo_12_2026-10-29_13-48-00") == "2026-10-29"
    assert resplit.session_of("5078") is None


def test_split_log_wins_over_session_rule():
    log = {"photo_1_2026-10-20_12-32-34": "test"}
    sessions = {"2026-10-20": "train"}
    assert resplit.target_split("photo_1_2026-10-20_12-32-34", log, sessions) == ("test", "split_log")
    assert resplit.target_split("photo_2_2026-10-20_12-32-34", log, sessions) == ("train", "session 2026-10-20")
    assert resplit.target_split("5078", log, sessions) == (None, None)


def test_parse_session_rejects_bad_split():
    assert resplit.parse_session("2026-10-29=valid") == ("2026-10-29", "valid")
    with pytest.raises(Exception):
        resplit.parse_session("2026-10-29=val")


def test_images_move_with_their_labels(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    make_export(src, {
        "train": ["5078_jpg.rf.aa.jpg", "photo_3_2026-10-29_13-48-00_jpg.rf.bb.jpg"],
        "test": ["photo_4_2026-10-29_13-48-00_jpg.rf.cc.jpg"],
    })
    log = tmp_path / "split_log.csv"
    log.write_text("file,new_split,old_split,cluster\n5078_jpg.rf.zz.jpg,valid,train,0\n")

    assert resplit.main([str(src), "--out", str(out), "--split-log", str(log),
                         "--session", "2026-10-29=train"]) == 0
    assert (out / "valid" / "images" / "5078_jpg.rf.aa.jpg").is_file()
    assert (out / "valid" / "labels" / "5078_jpg.rf.aa.txt").is_file()
    assert (out / "train" / "labels" / "photo_4_2026-10-29_13-48-00_jpg.rf.cc.txt").is_file()
    assert not (out / "test").exists()
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == ["person_left", "person_right"]
    with open(out / "resplit_log.csv") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_unmatched_images_stop_the_run(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    make_export(src, {"train": ["photo_1_2026-11-02_10-00-00_jpg.rf.aa.jpg"]})
    assert resplit.main([str(src), "--out", str(out), "--session", "2026-10-29=train"]) == 1
    assert not out.exists()


# --- --ratio: burst groups shared out at 70/20/10 ---

def noise_image(path, seed):
    """A random-looking image; different seeds give hashes far apart."""
    import random
    rng = random.Random(seed)
    img = Image.new("L", (32, 32))
    img.putdata([rng.randrange(256) for _ in range(32 * 32)])
    img.convert("RGB").resize((128, 128)).save(path)


def test_parse_ratio():
    assert resplit.parse_ratio("70/20/10") == [0.7, 0.2, 0.1]
    with pytest.raises(Exception):
        resplit.parse_ratio("70/30")


def test_burst_groups_chain_near_hashes():
    import imagehash
    import numpy as np
    base = np.zeros((8, 8), dtype=bool)
    near = base.copy(); near[0, :4] = True          # 4 bits from base
    nearer = near.copy(); nearer[1, :4] = True      # 4 bits from near, 8 from base
    far = ~base
    hashes = [imagehash.ImageHash(a) for a in (base, near, nearer, far)]
    assert resplit.burst_groups(hashes, 6) == [[0, 1, 2], [3]]


def test_share_out_hits_ratio():
    groups = [[i] for i in range(10)]
    got = resplit.share_out(groups, len, [0.7, 0.2, 0.1], dict.fromkeys(resplit.SPLITS, 0))
    from collections import Counter
    assert Counter(got.values()) == {"train": 7, "valid": 2, "test": 1}


def test_ratio_keeps_copies_together_and_spreads_bikes(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "data.yaml").write_text(yaml.safe_dump({"nc": 3, "names": ["person_left", "person_right", "bicycle"]}))
    for split in ("train", "valid"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
    # 20 distinct photos, 10 with a bike, plus an exact copy of photo 0 in valid
    for n in range(20):
        name = f"photo_{n}_2026-11-02_11-24-00_jpg.rf.x{n}"
        noise_image(src / "train" / "images" / f"{name}.jpg", n)
        lines = "0 0.3 0.5 0.1 0.3\n" + ("2 0.6 0.6 0.2 0.2\n" if n % 2 == 0 else "")
        (src / "train" / "labels" / f"{name}.txt").write_text(lines)
    noise_image(src / "valid" / "images" / "photo_99_2026-11-02_11-24-00_jpg.rf.copy.jpg", 0)
    (src / "valid" / "labels" / "photo_99_2026-11-02_11-24-00_jpg.rf.copy.txt").write_text("1 0.5 0.5 0.1 0.3\n")

    assert resplit.main([str(src), "--out", str(out), "--ratio", "70/20/10"]) == 0
    with open(out / "resplit_log.csv") as f:
        rows = {r["key"]: r for r in csv.DictReader(f)}
    assert rows["photo_0_2026-11-02_11-24-00_jpg"]["new_split"] == rows["photo_99_2026-11-02_11-24-00_jpg"]["new_split"]
    bike_splits = [rows[f"photo_{n}_2026-11-02_11-24-00_jpg"]["new_split"] for n in range(0, 20, 2)]
    assert bike_splits.count("train") == 7 and bike_splits.count("valid") == 2 and bike_splits.count("test") == 1


def test_ratio_and_split_log_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        resplit.main([str(tmp_path), "--out", str(tmp_path / "o"), "--ratio", "70/20/10", "--session", "2026-11-02=train"])
