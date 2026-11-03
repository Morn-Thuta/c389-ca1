"""Tests for src/data/verify_export.py and src/data/dedupe_check.py on synthetic datasets."""

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data import dedupe_check, verify_export


def noise_image(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)


def make_dataset(root, names=("person_left", "person_right"), labels=None):
    """Build root/data.yaml + train/valid/test with one noise image each.

    labels: {split: label file text}; defaults to one valid box per image.
    """
    labels = labels or {}
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": len(names), "names": list(names)}))
    for seed, split in enumerate(["train", "valid", "test"]):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        noise_image(root / split / "images" / f"{split}_0.jpg", seed)
        text = labels.get(split, "0 0.5 0.5 0.2 0.4\n")
        (root / split / "labels" / f"{split}_0.txt").write_text(text)
    return root


# --- verify_export ---

def test_clean_dataset_passes(tmp_path):
    make_dataset(tmp_path)
    report = verify_export.verify(tmp_path)
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["splits"]["train"]["boxes"]["person_left"] == 1
    assert verify_export.main([str(tmp_path)]) == 0


def test_four_class_order_passes(tmp_path):
    make_dataset(tmp_path, names=("person_left", "person_right", "wheeled_left", "wheeled_right"),
                 labels={"train": "3 0.5 0.5 0.3 0.3\n"})
    report = verify_export.verify(tmp_path, expect=4)
    assert report["errors"] == []
    assert report["splits"]["train"]["boxes"]["wheeled_right"] == 1


@pytest.mark.parametrize("names", [
    ("person_right", "person_left"),
    ("person_left", "person_right", "bicycle"),
    ("person_left", "person_right", "wheeled_right", "wheeled_left"),
    ("person",),
])
def test_wrong_class_order_fails(tmp_path, names):
    make_dataset(tmp_path, names=names)
    assert verify_export.main([str(tmp_path)]) == 1


def test_expect_mismatch_fails(tmp_path):
    make_dataset(tmp_path)
    assert verify_export.verify(tmp_path, expect=3)["errors"]


def test_out_of_range_id_fails(tmp_path):
    make_dataset(tmp_path, labels={"train": "2 0.5 0.5 0.2 0.4\n"})
    report = verify_export.verify(tmp_path)
    assert report["splits"]["train"]["bad_ids"]
    assert verify_export.main([str(tmp_path)]) == 1


def test_malformed_line_fails(tmp_path):
    make_dataset(tmp_path, labels={"valid": "0 0.5 0.5 0.2\n"})
    assert verify_export.verify(tmp_path)["splits"]["valid"]["malformed"]


def test_empty_and_small_boxes_are_warnings(tmp_path):
    make_dataset(tmp_path, labels={"train": "", "valid": "1 0.5 0.5 0.05 0.05\n"})
    report = verify_export.verify(tmp_path)
    assert report["errors"] == []
    assert report["splits"]["train"]["empty_labels"] == ["train_0"]
    assert report["splits"]["valid"]["small_boxes"]
    assert verify_export.main([str(tmp_path)]) == 0
    assert verify_export.main([str(tmp_path), "--strict"]) == 1


def test_missing_label_file_is_warning(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "test" / "labels" / "test_0.txt").unlink()
    assert verify_export.verify(tmp_path)["splits"]["test"]["missing_labels"] == ["test_0"]


def test_no_data_yaml_fails(tmp_path):
    assert verify_export.main([str(tmp_path)]) == 1


# --- dedupe_check ---

def test_no_duplicates_passes(tmp_path):
    make_dataset(tmp_path)
    pairs, counts = dedupe_check.find_cross_split_duplicates(tmp_path)
    assert pairs == []
    assert counts == {"train": 1, "valid": 1, "test": 1}
    assert dedupe_check.main([str(tmp_path)]) == 0


def test_cross_split_duplicate_fails(tmp_path):
    make_dataset(tmp_path)
    # same scene as train_0, re-saved slightly smaller (like a burst shot / re-export)
    with Image.open(tmp_path / "train" / "images" / "train_0.jpg") as img:
        img.resize((60, 60)).save(tmp_path / "test" / "images" / "test_1.jpg")
    pairs, _ = dedupe_check.find_cross_split_duplicates(tmp_path)
    assert [(p["image_a"], p["image_b"]) for p in pairs] == [("train_0.jpg", "test_1.jpg")]
    assert dedupe_check.main([str(tmp_path), "--csv", str(tmp_path / "out.csv")]) == 1
    assert (tmp_path / "out.csv").is_file()


def test_same_split_duplicate_is_ignored(tmp_path):
    make_dataset(tmp_path)
    with Image.open(tmp_path / "train" / "images" / "train_0.jpg") as img:
        img.save(tmp_path / "train" / "images" / "train_1.jpg")
    assert dedupe_check.main([str(tmp_path)]) == 0
