"""Tests for src/data/prelabel.py: merging, direction matching and file output (no models)."""

import csv

import yaml
from PIL import Image

from src.data import prelabel
from src.data.prelabel import PERSON_LEFT, PERSON_RIGHT, WHEELED_LEFT, WHEELED_RIGHT, match_direction, merge


def test_existing_labels_are_kept_and_duplicates_dropped():
    existing = [(PERSON_RIGHT, [100, 100, 200, 400])]
    proposals = [
        (PERSON_LEFT, [105, 98, 205, 405], "match"),   # same person, other class: dropped
        (PERSON_LEFT, [400, 100, 480, 380], "match"),  # new person: added
        (WHEELED_RIGHT, [600, 150, 740, 400], "match"),  # new rider: added
    ]
    labels, added = merge(existing, proposals)
    assert labels[0] == existing[0]                    # human label untouched
    assert [c for c, _, _ in added] == [PERSON_LEFT, WHEELED_RIGHT]
    assert len(labels) == 3


def test_box_inside_existing_label_is_a_duplicate():
    existing = [(PERSON_LEFT, [100, 100, 300, 500])]
    _, added = merge(existing, [(PERSON_LEFT, [120, 110, 200, 300], "match")])
    assert added == []


def test_person_inside_existing_wheeled_box_is_a_duplicate():
    existing = [(WHEELED_RIGHT, [80, 0, 220, 300])]
    _, added = merge(existing, [(PERSON_RIGHT, [100, 0, 200, 300], "match")])
    assert added == []


def test_proposals_do_not_duplicate_each_other():
    proposals = [(WHEELED_LEFT, [0, 0, 100, 100], "match"), (WHEELED_LEFT, [2, 2, 101, 99], "match")]
    _, added = merge([], proposals)
    assert len(added) == 1


def test_match_direction_picks_best_overlap():
    person = [100, 100, 200, 400]
    dets = [
        (PERSON_LEFT, 0.9, [300, 100, 400, 400]),   # no overlap
        (PERSON_RIGHT, 0.4, [102, 105, 198, 395]),  # best overlap
        (PERSON_LEFT, 0.6, [150, 100, 260, 400]),   # partial overlap
    ]
    assert match_direction(person, dets) == (PERSON_RIGHT, 0.4)


def test_match_direction_none_when_nothing_overlaps():
    assert match_direction([0, 0, 10, 10], [(PERSON_LEFT, 0.9, [50, 50, 60, 60])]) is None


def test_label_file_round_trip(tmp_path):
    path = tmp_path / "a.txt"
    labels = [(PERSON_LEFT, [10.0, 20.0, 110.0, 220.0]), (WHEELED_RIGHT, [200.0, 100.0, 400.0, 300.0])]
    prelabel.write_labels(path, labels, 640, 480)
    back = prelabel.read_labels(path, 640, 480)
    assert [c for c, _ in back] == [PERSON_LEFT, WHEELED_RIGHT]
    for (_, a), (_, b) in zip(labels, back):
        assert all(abs(x - y) < 0.01 for x, y in zip(a, b))


class FakeLabeller:
    def propose(self, image):
        return [
            (PERSON_RIGHT, [10, 10, 60, 120], "match"),
            (PERSON_LEFT, [100, 10, 150, 120], "direction_unknown"),
            (WHEELED_LEFT, [160, 10, 199, 120], "match"),
        ]


def test_run_writes_dataset_and_review_csv(tmp_path):
    src = tmp_path / "in"
    for split in ("train", "test"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
        Image.new("RGB", (200, 150)).save(src / split / "images" / f"{split}_0.jpg")
    # train already has the first person labelled by a human
    (src / "train" / "labels" / "train_0.txt").write_text("1 0.175 0.433 0.25 0.733\n")

    out = tmp_path / "out"
    rows, totals, _ = prelabel.run(src, out, FakeLabeller())

    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == ["person_left", "person_right", "wheeled_left", "wheeled_right"]
    train_labels = (out / "train" / "labels" / "train_0.txt").read_text().splitlines()
    test_labels = (out / "test" / "labels" / "test_0.txt").read_text().splitlines()
    assert len(train_labels) == 3      # 1 human + 2 added (duplicate person skipped)
    assert len(test_labels) == 3       # nothing existing, all 3 added
    assert (out / "test" / "images" / "test_0.jpg").is_file()
    assert totals["existing"] == 1
    assert totals["direction_unknown"] == 2

    with open(out / "prelabel_review.csv") as f:
        review = list(csv.DictReader(f))
    assert {r["image"] for r in review} == {"train_0.jpg", "test_0.jpg"}


def test_refuses_output_inside_repo(tmp_path):
    import pytest
    repo = prelabel.Path(prelabel.__file__).resolve().parents[2]
    with pytest.raises(SystemExit):
        prelabel.main([str(tmp_path), "--out", str(repo / "tmp_out"), "--direction-weights", "x.pt"])


# --- alphabetical Roboflow exports and the v2 split log ---

def make_alphabetical_export(root):
    """A Roboflow-style export: classes alphabetical, so person_left = 0, wheeled_right = 3."""
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 3, "names": ["person_right", "wheeled_left", "person_left"]}))
    for name in ("9_jpeg.rf.aa.jpg", "1000396141_jpg.rf.bb.jpg"):
        Image.new("RGB", (200, 150)).save(root / "train" / "images" / name)
    # one person_right (their id 0) and one wheeled_left (their id 1)
    (root / "train" / "labels" / "9_jpeg.rf.aa.txt").write_text(
        "0 0.175 0.433 0.25 0.733\n1 0.8 0.8 0.2 0.2\n")


class NoProposals:
    def propose(self, image):
        return []


def test_image_key_matches_reuploads_but_keeps_different_photos_apart():
    from src.data.prelabel import image_key
    assert image_key("9_jpeg.rf.ab12.jpg") == image_key("9_jpeg_jpg.rf.cd34.jpg") == "9_jpeg"
    assert image_key("2_jpg.rf.x.jpg") != image_key("2_jpeg.rf.y.jpg")
    assert image_key("photo_4_2026-10-20_12-32-34_jpg.rf.z.jpg") == "photo_4_2026-10-20_12-32-34_jpg"


def test_existing_labels_are_remapped_by_name(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    make_alphabetical_export(src)
    prelabel.run(src, out, NoProposals())
    lines = (out / "train" / "labels" / "9_jpeg.rf.aa.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in lines] == [str(PERSON_RIGHT), str(WHEELED_LEFT)]


def test_split_log_moves_images_and_drops_unlisted(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    make_alphabetical_export(src)
    log = tmp_path / "split_log.csv"
    log.write_text("file,new_split,old_split,cluster\n9_jpeg_jpg.rf.zz.jpg,test,train,0\n")
    rows, _, skipped = prelabel.run(src, out, NoProposals(), prelabel.read_split_log(log))
    assert skipped == ["1000396141_jpg.rf.bb.jpg"]
    assert [r["split"] for r in rows] == ["test"]
    assert (out / "test" / "images" / "9_jpeg.rf.aa.jpg").is_file()
    assert not (out / "train").exists()


def test_unknown_input_class_is_refused(tmp_path):
    import pytest
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["person_left", "person_other"]}))
    with pytest.raises(ValueError):
        prelabel.input_mapping(tmp_path)
