"""Service tests (PRD §9): bad inputs, empty scene, normal image, reply schema.

A fake detector stands in for the YOLO model, so these run without weights or
real photos. The live check against best.pt is done by hand (see the PR).
"""

import pytest
from PIL import Image

from app import service
from src.counting import COUNT_KEYS


class FakeDetector:
    mode = "fake"

    def __init__(self, detections=()):
        self.detections = list(detections)

    def __call__(self, image):
        return [dict(d) for d in self.detections]


RIDER_SCENE = [
    {"class_name": "wheeled_right", "confidence": 0.88, "box": [80, 0, 220, 300]},
    {"class_name": "person_left", "confidence": 0.75, "box": [300, 0, 380, 300]},
]


@pytest.fixture
def files(tmp_path):
    text = tmp_path / "not_an_image.txt"
    text.write_text("this is not a photograph")
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    huge = tmp_path / "huge.png"
    Image.new("L", (9000, 7000)).save(huge)  # 63 MP, small on disk
    normal = tmp_path / "scene.jpg"
    Image.new("RGB", (640, 480), (200, 200, 200)).save(normal)
    return {"text": text, "empty": empty, "huge": huge, "normal": normal}


@pytest.mark.parametrize("case, status", [
    ("nothing", 400),
    ("text", 415),
    ("empty", 400),
    ("huge", 413),
])
def test_bad_inputs_are_refused(files, case, status):
    uploaded = None if case == "nothing" else str(files[case])
    reply, image = service.build_reply(uploaded, detector=FakeDetector())
    assert image is None
    assert reply["status"] == status
    assert reply["error"] and reply["detail"]


def test_oversized_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "MAX_FILE_BYTES", 10)
    path = tmp_path / "small.jpg"
    Image.new("RGB", (64, 64)).save(path)
    reply, _ = service.build_reply(str(path), detector=FakeDetector())
    assert reply["status"] == 413


def test_empty_scene_returns_200_with_zeros(files):
    reply, image = service.build_reply(str(files["normal"]), detector=FakeDetector())
    assert reply["status"] == 200
    assert image is not None
    assert all(reply["counts"][k] == 0 for k in COUNT_KEYS)


def test_normal_image_counts_and_schema(files):
    reply, _ = service.build_reply(str(files["normal"]), 0.4, detector=FakeDetector(RIDER_SCENE))
    assert reply["status"] == 200
    assert list(reply["counts"]) == COUNT_KEYS
    assert reply["counts"]["wheeled_right"] == 1
    assert reply["counts"]["person_left"] == 1
    assert reply["counts"]["total_left"] == 1 and reply["counts"]["total_right"] == 1
    for key in ("status", "counts", "confidence_threshold", "model_version",
                "model_mode", "image_size", "inference_seconds", "detections"):
        assert key in reply


def test_accepts_bytes_and_pil(files):
    data = files["normal"].read_bytes()
    assert service.build_reply(data, detector=FakeDetector())[0]["status"] == 200
    assert service.build_reply(Image.open(files["normal"]), detector=FakeDetector())[0]["status"] == 200
    assert service.build_reply(b"", detector=FakeDetector())[0]["status"] == 400


def test_predict_returns_annotated_image(files, monkeypatch):
    monkeypatch.setattr(service, "DETECTOR", FakeDetector(RIDER_SCENE))
    marked, reply = service.predict(str(files["normal"]), 0.4)
    assert reply["status"] == 200
    assert marked.size == (640, 480)


def test_predict_on_refusal_returns_no_image(files, monkeypatch):
    monkeypatch.setattr(service, "DETECTOR", FakeDetector())
    marked, reply = service.predict(str(files["text"]), 0.4)
    assert marked is None
    assert reply["status"] == 415
