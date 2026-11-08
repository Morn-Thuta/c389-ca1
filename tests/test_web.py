"""Tests for the web API (app/web.py) with a fake detector: no weights, no photos."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import service, web
from src.counting import COUNT_KEYS


class FakeDetector:
    mode = "fake"
    weights = "fake.pt"

    def __call__(self, image):
        w, h = image.size
        return [
            {"class_name": "person_left", "confidence": 0.9, "box": [0.1 * w, 0.2 * h, 0.2 * w, 0.8 * h]},
            {"class_name": "wheeled_right", "confidence": 0.8, "box": [0.7 * w, 0.2 * h, 0.9 * w, 0.9 * h]},
            {"class_name": "person_right", "confidence": 0.2, "box": [0.4 * w, 0.2 * h, 0.5 * w, 0.8 * h]},
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "DETECTOR", FakeDetector())
    return TestClient(web.app)


def jpeg(size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, "JPEG")
    return buf.getvalue()


def post(client, data=None, **form):
    files = {"image": ("scene.jpg", data, "image/jpeg")} if data is not None else None
    return client.post("/api/count", files=files, data=form)


def test_page_and_health(client):
    page = client.get("/")
    assert page.status_code == 200 and "People Counting Assistant" in page.text
    health = client.get("/api/health").json()
    assert health["model_loaded"] and health["count_keys"] == COUNT_KEYS


def test_counts_whole_image(client):
    res = post(client, jpeg())
    body = res.json()
    assert res.status_code == 200 and body["status"] == 200
    assert body["counts"]["person_left"] == 1 and body["counts"]["wheeled_right"] == 1
    assert body["counts"]["person_right"] == 0           # 0.2 is under the 0.40 default
    assert body["roi"] is None and body["outside_roi"] == 0
    assert list(body["counts"]) == COUNT_KEYS


def test_threshold_is_applied(client):
    body = post(client, jpeg(), threshold="0.1").json()
    assert body["counts"]["person_right"] == 1 and body["confidence_threshold"] == 0.1


def test_roi_counts_left_half_only(client):
    roi = json.dumps([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])
    body = post(client, jpeg(), roi=roi).json()
    assert body["counts"]["person_left"] == 1 and body["counts"]["wheeled_right"] == 0
    assert body["outside_roi"] == 1 and body["roi"][1] == [0.5, 0.0]


@pytest.mark.parametrize("form, status", [
    ({"roi": "not json"}, 422),
    ({"roi": "[[0, 0], [1, 1]]"}, 422),
    ({"threshold": "abc"}, 422),
    ({"threshold": "1.5"}, 422),
])
def test_bad_settings_are_refused(client, form, status):
    res = post(client, jpeg(), **form)
    assert res.status_code == status and res.json()["error"]


def test_bad_uploads_are_refused(client):
    assert post(client).status_code == 400                       # nothing sent
    assert post(client, b"").status_code == 400                  # empty file
    assert post(client, b"this is not a photograph").status_code == 415


def test_no_model_is_503(monkeypatch):
    monkeypatch.setattr(service, "DETECTOR", None)
    res = TestClient(web.app).post("/api/count", files={"image": ("a.jpg", jpeg(), "image/jpeg")})
    assert res.status_code == 503


def test_sideways_phone_photo_is_turned_upright(client):
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6                                              # "rotate 90° clockwise to view"
    Image.new("RGB", (320, 240)).save(buf, "JPEG", exif=exif)
    assert post(client, buf.getvalue()).json()["image_size"] == [240, 320]
