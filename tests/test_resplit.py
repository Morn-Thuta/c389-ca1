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
    assert resplit.main([str(src), "--out", str(out)]) == 1
    assert not out.exists()
