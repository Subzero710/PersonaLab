"""Published weighted locations, explicit aliases and legacy serialization."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from persona_lab.benchmark.model import MODEL_SHA256
from persona_lab.benchmark.storage import digest_arrays, write_json
from persona_lab.dense_observations import ObservationPoint, ObservationSchema
from persona_lab.facepoints_contract import FACEPOINTS_POINTS_SHA256, FACEPOINTS_CLASS_ORDER, FACEPOINTS_GROUPS
from persona_lab.surface_correspondences import (
    NativeLocation, SurfaceCorrespondence, SurfaceCorrespondences, load_correspondences,
)


def fixture_mapping():
    # Contract fixture only: no fabricated real GNM correspondence file.
    count = 630
    triangles = np.array([[i, i+1, i+2] for i in range(591)], dtype=np.int64)
    topology = digest_arrays({"triangles": triangles})
    groups = ("skin", "upper_teeth_and_gums", "lower_teeth_and_gums", "eye_interiors")
    model = SimpleNamespace(triangles=triangles, metadata={"asset_sha256":MODEL_SHA256,
        "topology_sha256":topology}, head=SimpleNamespace(triangles=torch.from_numpy(triangles),
        num_vertices=count, vertex_group_names=groups,
        vertex_groups=torch.ones((len(groups),count)), mirror_indices=torch.arange(count)))
    points, entries = [], []
    for i in range(620):
        identifier = f"fp:{i}"
        region = FACEPOINTS_CLASS_ORDER[i]
        group = FACEPOINTS_GROUPS[region]
        if i < 602:
            source = i if i < 591 else i - 591
            vertices, weights = (source, source+1, source+2), (.2,.3,.499999)
            alias = f"fp:{source}" if i > 590 else None
        elif i < 618:
            vertices, weights, alias = (i,), (1.,), None
        else:
            vertices, weights, alias = tuple(range(i-618, i-618+145)), (.006896552,)*145, None
        points.append(ObservationPoint(identifier, region))
        entries.append(SurfaceCorrespondence(identifier, NativeLocation("weighted_vertices", None, weights, vertices),
                        (group,), "synthetic contract fixture", "not a real mapping", duplicate_of=alias))
    schema = ObservationSchema("facepoints-620-dfl",1,"aligned_pixels",tuple(points))
    return model, SurfaceCorrespondences(schema, topology, tuple(entries), FACEPOINTS_POINTS_SHA256)


def test_weighted_roundtrip_projection_and_gradient(tmp_path):
    model, definition = fixture_mapping()
    path = tmp_path / "mapping.json"
    write_json(path, definition.to_record())
    assert load_correspondences(path, definition.schema) == definition
    bound = definition.bind(model)
    assert bound.indices.shape == bound.weights.shape == (620,145)
    vertices = (torch.arange(2*630*3, dtype=torch.float64).reshape(2,630,3) / 10000).requires_grad_()
    actual = bound.evaluate(vertices)
    direct = torch.stack([(vertices[...,list(e.location.vertex_indices),:] *
                            vertices.new_tensor(e.location.weights)[...,None]).sum(-2)
                          for e in definition.entries],dim=-2)
    torch.testing.assert_close(actual, direct, rtol=0, atol=2e-12)
    probe = torch.linspace(.1,1.,actual.numel(),dtype=actual.dtype).reshape_as(actual)
    actual_grad, = torch.autograd.grad(actual, vertices, probe)
    direct_grad, = torch.autograd.grad(direct, vertices, probe)
    torch.testing.assert_close(actual_grad, direct_grad, rtol=0, atol=2e-12)
    assert torch.equal(actual[:,10],actual[:,601])
    assert np.array_equal(bound.weights[0,:3], [.2,.3,.499999])
    assert bound.weights[0,3:].sum() == 0


def test_alias_and_wrong_topology_are_rejected():
    model, definition = fixture_mapping()
    entries = list(definition.entries)
    entries[591] = replace(entries[591], duplicate_of=None)
    with pytest.raises(ValueError, match="duplicates"):
        replace(definition, entries=tuple(entries)).bind(model)
    with pytest.raises(ValueError, match="topology"):
        replace(definition, topology_sha256="0"*64).bind(model)
    with pytest.raises(ValueError, match="weights"):
        NativeLocation("weighted_vertices", None, (.2,.3,.49), (0,1,2))
    with pytest.raises(ValueError, match="weights"):
        NativeLocation("triangle", 0, (.2,.3,.499999))


def test_legacy_record_keeps_original_fields(tmp_path):
    schema = ObservationSchema("legacy",1,"source_pixels",(ObservationPoint("a","skin"),))
    definition = SurfaceCorrespondences(schema,"0"*64,(SurfaceCorrespondence(
        "a",NativeLocation("triangle",0,(.2,.3,.5)),("skin",),"fixture","fixture"),))
    record = definition.to_record()
    assert record["format"] == "gnm-surface-correspondences-v1"
    assert set(record["entries"][0]["location"]) == {"kind","index","weights"}
    assert "duplicate_of" not in record["entries"][0]
    path = tmp_path / "legacy.json"
    write_json(path,record)
    assert load_correspondences(path,schema) == definition
