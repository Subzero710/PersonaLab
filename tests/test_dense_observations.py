"""Tests for generalized schema and numeric dataset integrity."""

from dataclasses import replace

import pytest
import torch

from persona_lab.coordinates import ImageSize
from persona_lab.dense_dataset import ExtractionAttempt, load_dense_dataset, write_dense_dataset
from persona_lab.dense_observations import (
    DenseObservation, ImageProvenance, ModelAsset, ObservationPoint, ObservationSchema,
)
from persona_lab.observations import AlignmentMetadata


@pytest.fixture
def observation():
    size = ImageSize(100, 80)
    schema = ObservationSchema("three", 2, "source_pixels", tuple(
        ObservationPoint(f"point:{index}", "test-region") for index in range(3)
    ))
    return DenseObservation(
        observation_id="raster", schema=schema, provider="test-provider", provider_version="1.2",
        asset=ModelAsset("one", "test://asset", "f" * 64),
        source=ImageProvenance("raster.jpg", "a" * 64, size),
        extraction={"device": "cpu", "mode": "image"}, image_size=size,
        landmarks=torch.tensor([[30., 12.], [30., 23.], [0., 0.]], dtype=torch.float64),
        validity=torch.tensor([True, True, False]),
        visibility=torch.tensor([0.3, -1., -1.], dtype=torch.float64),
    )


def test_dense_metadata_and_round_trip(tmp_path, observation):
    obs = observation
    directory = tmp_path / "dense"
    write_dense_dataset(directory, obs.schema, obs.provider, obs.provider_version, obs.asset,
                        obs.extraction, (
        ExtractionAttempt(obs.observation_id, obs.source.reference, obs, image_sha256=obs.source.sha256),
        ExtractionAttempt("failed", "invalid.jpg", None, "decode failed"),
    ))
    metadata = dict(schema=obs.schema, provider=obs.provider, provider_version=obs.provider_version,
                    asset=obs.asset)
    data = load_dense_dataset(directory, **metadata)
    returned, = data.select(["raster"])
    assert returned.schema.point_ids == obs.schema.point_ids
    assert returned.source == obs.source
    assert returned.confidence is None
    assert torch.equal(returned.visibility, obs.visibility)
    assert torch.equal(returned.validity, obs.validity)
    assert torch.equal(returned.landmarks, obs.landmarks)
    with pytest.raises(ValueError, match="failed observations"):
        data.select()
    with pytest.raises(ValueError, match="precision|float64"):
        load_dense_dataset(directory, **metadata, dtype=torch.float32)
    with pytest.raises(ValueError, match="incompatible"):
        load_dense_dataset(directory, **(metadata | {"schema": replace(obs.schema, version=3)}))


def test_schema_and_coordinates(observation):
    obs = observation
    with pytest.raises(ValueError, match="requires its SHA-256"):
        replace(obs.source, sha256=None)
    with pytest.raises(ValueError, match="unique"):
        replace(obs.schema, points=(obs.schema.points[0],) * 2)
    with pytest.raises(ValueError, match="shape"):
        replace(obs, landmarks=obs.landmarks[:2])
    with pytest.raises(ValueError, match="nonfinite|finite"):
        replace(obs, landmarks=obs.landmarks * float("nan"))
    with pytest.raises(ValueError, match="unavailable"):
        replace(obs, visibility=obs.landmarks.new_full((3,), -1.))
    cropped = replace(
        obs, schema=replace(obs.schema, coordinate_convention="aligned_pixels"),
        source=replace(obs.source, source_image_size=None),
        alignment=AlignmentMetadata(image_to_face_mat=obs.landmarks.new_tensor([[2, 0, 4], [0, 2, 2]])),
    )
    torch.testing.assert_close(cropped.source_pixels(), (obs.landmarks - obs.landmarks.new_tensor([4, 2])) / 2)
    with pytest.raises(ValueError, match="image_to_face_mat"):
        replace(cropped, alignment=None).source_pixels()
