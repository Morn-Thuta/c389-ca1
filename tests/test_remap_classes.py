"""Tests for src/data/remap_classes.py."""

import pytest
import yaml

from src.data import remap_classes


def make_export(root, names, labels):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": len(names), "names": names}))
    (root / "train" / "images" / "a.jpg").write_bytes(b"jpg")
    (root / "train" / "labels" / "a.txt").write_text(labels)


def test_alphabetical_export_is_put_in_our_order(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    make_export(src, ["bicycle", "person_left", "person_right"],
                "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.3\n2 0.8 0.8 0.1 0.3\n")
    mapping, count = remap_classes.remap_dataset(src, out)
    assert mapping == {0: 2, 1: 0, 2: 1}
    assert count == 1
    ids = [ln.split()[0] for ln in (out / "train" / "labels" / "a.txt").read_text().splitlines()]
    assert ids == ["2", "0", "1"]
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == ["person_left", "person_right", "bicycle"]
    assert (out / "train" / "images" / "a.jpg").is_file()


def test_two_class_export_stays_two_classes(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    make_export(src, ["person_left", "person_right"], "1 0.5 0.5 0.1 0.3\n")
    mapping, _ = remap_classes.remap_dataset(src, out)
    assert mapping == {0: 0, 1: 1}
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == ["person_left", "person_right"]


def test_unknown_class_is_refused(tmp_path):
    src = tmp_path / "in"
    make_export(src, ["person", "bicycle"], "")
    with pytest.raises(ValueError):
        remap_classes.remap_dataset(src, tmp_path / "out")


def test_cli_refuses_same_folder(tmp_path):
    with pytest.raises(SystemExit):
        remap_classes.main([str(tmp_path), "--out", str(tmp_path)])
