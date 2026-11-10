"""Tests for src/data/to_wheeled.py: bicycle exports -> wheeled_left / wheeled_right."""

import csv

import pytest
import yaml

from src.data import to_wheeled


def make_export(root, labels):
    """Alphabetical Roboflow export: bicycle 0, person_left 1, person_right 2."""
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 3, "names": ["bicycle", "person_left", "person_right"]}))
    for name, text in labels.items():
        (root / "train" / "images" / f"{name}.jpg").write_bytes(b"jpg")
        (root / "train" / "labels" / f"{name}.txt").write_text(text)


def boxes(path):
    return [(int(ln.split()[0]), [round(float(v), 3) for v in ln.split()[1:]])
            for ln in path.read_text().splitlines()]


def test_rider_becomes_one_wheeled_box(tmp_path):
    # person_right at x 0.10-0.20, y 0-0.30; bike under the legs at x 0.08-0.22, y 0.18-0.30
    make_export(tmp_path / "in", {"ride": "2 0.15 0.15 0.10 0.30\n0 0.15 0.24 0.14 0.12\n"})
    rows, totals = to_wheeled.convert_dataset(tmp_path / "in", tmp_path / "out")
    assert boxes(tmp_path / "out" / "train" / "labels" / "ride.txt") == [(3, [0.15, 0.15, 0.14, 0.30])]
    assert totals["riders_merged"] == 1 and totals["bikes_removed"] == 0
    assert rows == [{"image": "ride", "split": "train", "riders_merged": 1, "bikes_removed": 0}]


def test_unridden_bike_is_removed_and_walker_kept(tmp_path):
    # person_left on the left, a parked bike far to the right
    make_export(tmp_path / "in", {"park": "1 0.15 0.5 0.10 0.30\n0 0.80 0.6 0.14 0.12\n"})
    _, totals = to_wheeled.convert_dataset(tmp_path / "in", tmp_path / "out")
    assert boxes(tmp_path / "out" / "train" / "labels" / "park.txt") == [(0, [0.15, 0.5, 0.10, 0.30])]
    assert totals["bikes_removed"] == 1 and totals["person_left"] == 1


def test_output_is_in_our_class_order_with_review_csv(tmp_path):
    make_export(tmp_path / "in", {"walk": "2 0.5 0.5 0.1 0.3\n"})
    to_wheeled.convert_dataset(tmp_path / "in", tmp_path / "out")
    assert yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())["names"] == [
        "person_left", "person_right", "wheeled_left", "wheeled_right"]
    assert (tmp_path / "out" / "train" / "images" / "walk.jpg").is_file()
    with open(tmp_path / "out" / "to_wheeled_review.csv") as f:
        assert list(csv.DictReader(f)) == []  # nothing merged or removed


def test_unknown_class_is_refused(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "data.yaml").write_text(yaml.safe_dump({"names": ["person", "car"]}))
    with pytest.raises(ValueError):
        to_wheeled.convert_dataset(tmp_path / "in", tmp_path / "out")
