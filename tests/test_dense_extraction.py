"""Raster extraction contracts and optional real MediaPipe CPU integration."""

import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from persona_lab.coordinates import ImageSize
from persona_lab.dense_dataset import load_dense_dataset
from persona_lab.extract_dense import _selected, extract_dataset
from persona_lab.mediapipe_face import ASSET, PROVIDER, PROVIDER_VERSION, SCHEMAS, MediaPipeFaceLandmarker, as_pixels


def _config(asset_path, **overrides):
    return dict(input_kind="images", schema_name="mediapipe-face-478", provider=PROVIDER, device="cpu",
                asset_version=ASSET.version, asset_path=asset_path, batch_size=1) | overrides


def test_provider_requires_explicit_device_and_asset(tmp_path):
    absent = tmp_path / "model.task"
    with pytest.raises(ValueError, match="CPU"):
        MediaPipeFaceLandmarker(absent, asset_version=ASSET.version, device="cuda", score_threshold=0.5)
    with pytest.raises(ValueError, match="Cannot open"):
        MediaPipeFaceLandmarker(absent, asset_version=ASSET.version, device="cpu", score_threshold=0.5)
    absent.write_bytes(b"unexpected weights")
    with pytest.raises(ValueError, match="SHA-256"):
        MediaPipeFaceLandmarker(absent, asset_version=ASSET.version, device="cpu", score_threshold=0.5)


def test_provider_pixels_use_original_raster_and_separate_scores():
    points = [SimpleNamespace(x=0.3, y=0.5, visibility=None, presence=None) for _ in range(478)]
    points[0] = SimpleNamespace(x=1.1, y=0.5, visibility=0.2, presence=None)
    output = as_pixels(points, ImageSize(200, 100))
    torch.testing.assert_close(output["landmarks"][1], torch.tensor([60., 50.], dtype=torch.float64))
    assert not output["validity"][0] and output["validity"][1:].all()
    assert output["visibility"][:2].tolist() == [0.2, -1.]
    assert output["presence"] is None and "confidence" not in output


def test_dfl_zip_attempt_order_and_failures(tmp_path, dfl_jpeg, monkeypatch):
    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract(self, image, size):
            return {
                "landmarks": torch.ones((478, 2), dtype=torch.float64),
                "validity": torch.ones(478, dtype=torch.bool),
            }

    monkeypatch.setattr("persona_lab.extract_dense.MediaPipeFaceLandmarker", Stub)
    path = tmp_path / "fixture.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("b.jpg", dfl_jpeg())
        archive.writestr("a.jpg", b"invalid")
        archive.writestr("c.jpg", dfl_jpeg())
    settings = _config(tmp_path / "stub.task", input_kind="dfl", schema_name="mediapipe-face-478-dfl",
                       selection="sample", seed=44, max_images=3)
    a = extract_dataset(path, tmp_path / "batch1", overlays=1, **settings)
    b = extract_dataset(path, tmp_path / "batch3", **(settings | {"batch_size": 3}))
    assert a["attempted"] == b["attempted"] == 3
    assert a["succeeded"] == 2 and a["failed"] == 1
    assert a["attempts"][1]["dfl_valid"] == 68 and a["attempts"][1]["valid"] == 478
    dataset = load_dense_dataset(tmp_path / "batch1", schema=SCHEMAS["mediapipe-face-478-dfl"],
                                 provider=PROVIDER, provider_version=PROVIDER_VERSION, asset=ASSET)
    assert [attempt.observation_id for attempt in dataset.attempts] == ["zip:a.jpg", "zip:b.jpg", "zip:c.jpg"]
    assert dataset.attempts[0].error
    assert dataset.observations[0].source.source_image_size is None
    assert [x["observation_id"] for x in a["attempts"]] == [x["observation_id"] for x in b["attempts"]]
    explicit = settings | {"selection": "first", "max_images": None, "observation_ids": ("zip:c.jpg", "zip:a.jpg"),
                           "require_images": 2}
    c = extract_dataset(path, tmp_path / "explicit", **explicit)
    assert [row["observation_id"] for row in c["attempts"]] == ["zip:c.jpg", "zip:a.jpg"]
    assert (c["succeeded"], c["failed"]) == (1, 1)
    with pytest.raises(ValueError, match="required image count"):
        extract_dataset(path, tmp_path / "wrong-count", **(explicit | {"require_images": 3}))


def test_pinned_mediapipe_real_image(tmp_path):
    asset = os.environ.get("PERSONALAB_MEDIAPIPE_MODEL")
    image = os.environ.get("PERSONALAB_PUBLIC_PORTRAIT")
    if not asset or not image:
        pytest.skip("Set PERSONALAB_MEDIAPIPE_MODEL and PERSONALAB_PUBLIC_PORTRAIT for external CPU integration")
    info = extract_dataset(Path(image), tmp_path / "real", overlays=1, **_config(asset))
    assert (info["attempted"], info["succeeded"], info["failed"]) == (1, 1, 0)
    assert info["attempts"][0]["valid"] == 478
    assert info["attempts"][0]["mapped"] == 0
    assert info["attempts"][0]["score_slots"] == {"confidence": 0, "visibility": 0, "presence": 0}


def test_explicit_selection_preserves_order_and_rejects_missing():
    inputs = [SimpleNamespace(reference=name) for name in ("a.jpg", "b.jpg", "c.jpg")]
    assert [row.reference for row in _selected(inputs, None, "first", 42, ("c.jpg", "a.jpg"))] == [
        "c.jpg", "a.jpg",
    ]
    with pytest.raises(ValueError, match="Requested observations missing"):
        list(_selected(inputs, None, "first", 42, ("absent.jpg",)))
    with pytest.raises(ValueError, match="Duplicate requested observation ID"):
        list(_selected(inputs + inputs, None, "first", 42, ("a.jpg",)))
