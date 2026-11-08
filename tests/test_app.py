"""Tests for the Gradio page's functions (app/app.py) with a fake detector: no weights, no photos."""

import pytest
from PIL import Image

from app import app, service


class FakeDetector:
    mode = "fake"

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        w, h = image.size
        return [
            {"class_name": "person_left", "confidence": 0.9, "box": [0.1 * w, 0.2 * h, 0.2 * w, 0.8 * h]},
            {"class_name": "wheeled_right", "confidence": 0.8, "box": [0.7 * w, 0.2 * h, 0.9 * w, 0.9 * h]},
        ]


class Click:
    """Stands in for gr.SelectData: pixel position of a click on the Output photo."""

    def __init__(self, x, y):
        self.index = [x, y]


@pytest.fixture
def fake(monkeypatch):
    detector = FakeDetector()
    monkeypatch.setattr(service, "DETECTOR", detector)
    monkeypatch.setattr(app, "CACHED", None)
    return detector


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "scene.jpg"
    Image.new("RGB", (400, 200), (200, 200, 200)).save(path)
    return str(path)


def test_run_counts_whole_photo(fake, photo):
    marked, html, reply, state = app.run(photo, 0.4, [])
    assert marked.size == (400, 200)
    assert reply["counts"]["person_left"] == 1 and reply["counts"]["wheeled_right"] == 1
    assert reply is state and reply["roi"] is None
    assert "person_left:" in html and "wheeled_right:" in html


def test_no_photo_and_bad_file(fake, tmp_path):
    marked, html, reply, _ = app.run(None, 0.4, [])
    assert marked is None and reply is None and "Upload a photo" in html
    text = tmp_path / "notes.txt"
    text.write_text("not a photo")
    marked, html, reply, _ = app.run(str(text), 0.4, [])
    assert marked is None and reply["status"] == 415 and "not a readable image" in html


def test_three_clicks_make_an_area_and_the_model_runs_once(fake, photo):
    _, _, _, reply = app.run(photo, 0.4, [])
    corners = []
    # Left half of the photo: corners at the top and bottom of x = 0 and x = 200 px.
    for x, y in [(0, 0), (200, 0), (200, 200)]:
        corners, marked, html, reply, _, note = app.add_corner(photo, 0.4, corners, reply, Click(x, y))
    corners, marked, html, reply, _, note = app.add_corner(photo, 0.4, corners, reply, Click(0, 200))
    assert corners == [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    assert reply["counts"]["person_left"] == 1 and reply["counts"]["wheeled_right"] == 0
    assert reply["outside_roi"] == 1 and "4 corners" in note
    assert fake.calls == 1                       # clicks re-count from the cached detections


def test_undo_and_clear(fake, photo):
    corners = [[0, 0], [0.5, 0], [0.5, 1]]
    corners, _, _, reply, _, note = app.undo_corner(photo, 0.4, corners)
    assert len(corners) == 2 and reply["roi"] is None and "1 more" in note
    corners, _, _, reply, _, note = app.clear_area(photo, 0.4)
    assert corners == [] and reply["roi"] is None and "none" in note


def test_click_before_any_photo_changes_nothing(fake):
    out = app.add_corner(None, 0.4, [], None, Click(10, 10))
    assert out[0] == []


def test_new_photo_runs_the_model_again(fake, photo, tmp_path):
    app.run(photo, 0.4, [])
    other = tmp_path / "other.jpg"
    Image.new("RGB", (400, 200), (10, 10, 10)).save(other)
    app.run(str(other), 0.4, [])
    assert fake.calls == 2
