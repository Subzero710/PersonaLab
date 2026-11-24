"""Validate coordinates, optional fields and real NumPy pickle patterns."""

from io import BytesIO
import pickle
import struct
from zlib import crc32

import numpy as np
from PIL import Image
import pytest

from personalab.dfl import ErrorCode, SampleError, read_jpeg


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_numpy_serialization_preserves_source_and_alignment(dfl_file, dfl_points, protocol):
    aligned = np.asfortranarray(dfl_points, dtype=">f4")
    source = np.asarray(dfl_points + 19, dtype=np.float64)
    mat = np.array([[1.25, -0.2, 3.0], [0.1, 1.1, -4.0]], dtype=np.float32)
    raw_polys = {"polys": [{"type": 1, "pts": aligned[:3].copy()}]}
    path = dfl_file(
        {
            "landmarks": aligned, "face_type": "whole_face", "source_filename": "0007.png",
            "source_rect": (np.int32(-2), 3, 100, 121), "source_landmarks": source.tolist(),
            "image_to_face_mat": mat, "eyebrows_expand_mod": np.float32(1.2),
            "seg_ie_polys": raw_polys, "extra": {"tag": b"value", "empty": np.empty(0, np.uint8)},
        },
        protocol=protocol,
    )
    sample = read_jpeg(path)
    assert sample.landmarks.dtype == aligned.dtype
    assert sample.landmarks.shape == (68, 2)
    np.testing.assert_array_equal(sample.landmarks, aligned)
    np.testing.assert_array_equal(sample.source_landmarks, source)
    np.testing.assert_array_equal(sample.source_rect, [-2, 3, 100, 121])
    np.testing.assert_array_equal(sample.image_to_face_mat, mat)
    assert sample.image_to_face_mat.dtype == mat.dtype
    assert sample.face_type == "whole_face"
    assert sample.source_filename == "0007.png"
    assert sample.additional_metadata["eyebrows_expand_mod"] == np.float32(1.2)
    np.testing.assert_array_equal(
        sample.additional_metadata["seg_ie_polys"]["polys"][0]["pts"], aligned[:3],
    )
    assert sample.additional_metadata["extra"]["tag"] == b"value"
    assert sample.additional_metadata["extra"]["empty"].shape == (0,)


def test_numpy_legacy_module_name(dfl_file, dfl_points):
    payload = pickle.dumps({"landmarks": dfl_points}, protocol=3)
    payload = payload.replace(b"numpy._core.multiarray\n", b"numpy.core.multiarray\n")
    sample = read_jpeg(dfl_file(None, app15=(payload,)))
    np.testing.assert_array_equal(sample.landmarks, dfl_points)


@pytest.mark.parametrize("flags", [b"K\x00K\x01", b"K\x01K\x01", b"K\x00K\x02", b"K\x00\x88"])
def test_legacy_numpy_integer_dtype_flags(dfl_file, dfl_points, mask_buffer, flags):
    # Protocol 3 DFL files from older NumPy releases use BININT1 0, 1
    # instead of NEWFALSE, NEWTRUE in the dtype reducer arguments.
    rect = np.array([0, 0, 100, 120], dtype=np.int32)
    matrix = np.eye(2, 3, dtype=np.float64)
    mask = np.zeros((7, 9), dtype=np.uint8)
    payload = pickle.dumps({
        "landmarks": dfl_points.tolist(),
        "source_rect": rect,
        "image_to_face_mat": matrix,
        "xseg_mask": mask_buffer(mask),
    }, protocol=3)
    payload = payload.replace(b"numpy._core.multiarray\n", b"numpy.core.multiarray\n")
    assert payload.count(b"\x89\x88\x87") == 3
    payload = payload.replace(b"\x89\x88\x87", flags + b"\x87")
    path = dfl_file(None, app15=(payload,))
    if flags != b"K\x00K\x01":
        with pytest.raises(SampleError) as rejected:
            read_jpeg(path)
        assert rejected.value.code == ErrorCode.UNSUPPORTED_METADATA_TYPE
        return
    sample = read_jpeg(path)
    np.testing.assert_array_equal(sample.landmarks, dfl_points)
    np.testing.assert_array_equal(sample.source_rect, rect)
    np.testing.assert_array_equal(sample.image_to_face_mat, matrix)
    np.testing.assert_array_equal(sample.xseg_mask, mask[..., None])


def test_optional_fields_remain_absent(dfl_file):
    sample = read_jpeg(dfl_file())
    assert sample.face_type is sample.source_filename is sample.source_rect is None
    assert sample.source_landmarks is sample.image_to_face_mat is sample.xseg_mask is None
    assert sample.additional_metadata == {}


@pytest.mark.parametrize(
    ("landmarks", "code"),
    [
        (None, ErrorCode.MISSING_LANDMARKS),
        (np.zeros((67, 2)), ErrorCode.INVALID_LANDMARK_SHAPE),
        ([[1], [2, 3]], ErrorCode.INVALID_LANDMARK_SHAPE),
        ([["1", 2]] * 68, ErrorCode.INVALID_LANDMARK_TYPE),
        ([[True, 2]] * 68, ErrorCode.INVALID_LANDMARK_TYPE),
        (np.full((68, 2), np.nan), ErrorCode.NON_FINITE_LANDMARKS),
        ([[0.0, np.inf]] * 68, ErrorCode.NON_FINITE_LANDMARKS),
    ],
)
def test_invalid_landmarks(dfl_file, landmarks, code):
    metadata = {"face_type": "full_face"}
    if landmarks is not None:
        metadata["landmarks"] = landmarks
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file(metadata))
    assert rejected.value.code == code
    assert rejected.value.rejection.issues[0].field == "landmarks"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("image_to_face_mat", np.eye(3), ErrorCode.INVALID_TRANSFORM),
        ("image_to_face_mat", [[np.nan, 0, 0], [0, 1, 0]], ErrorCode.INVALID_TRANSFORM),
        ("image_to_face_mat", [["1", 0, 0], [0, 1, 0]], ErrorCode.INVALID_TRANSFORM),
        ("source_filename", 7, ErrorCode.INVALID_ALIGNMENT_METADATA),
        ("source_rect", (2, 0, 1, 4), ErrorCode.INVALID_ALIGNMENT_METADATA),
        ("seg_ie_polys", {"polys": [2]}, ErrorCode.INVALID_ALIGNMENT_METADATA),
        ("face_type", "mark_only", ErrorCode.UNSUPPORTED_METADATA_TYPE),
    ],
)
def test_present_invalid_optional_metadata_is_rejected(dfl_file, dfl_points, field, value, code):
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, field: value}))
    assert rejected.value.code == code
    assert rejected.value.rejection.issues[0].field == field
    assert field not in rejected.value.rejection.missing_optional_fields


@pytest.mark.parametrize("codec", ["PNG", "JPEG"])
def test_xseg_preserves_resolution_and_normalizes_grayscale(dfl_file, dfl_points, mask_buffer, codec):
    pixels = np.resize(np.array([0, 64, 128, 255], dtype=np.uint8), (11, 9))
    buffer = mask_buffer(pixels, codec)
    with Image.open(BytesIO(buffer.tobytes())) as mask:
        expected = np.asarray(mask, dtype=np.float32) / np.float32(255)
    sample = read_jpeg(dfl_file({"landmarks": dfl_points, "xseg_mask": buffer}))
    assert sample.xseg_mask.shape == (11, 9, 1)
    assert sample.xseg_mask.dtype == np.float32
    np.testing.assert_array_equal(sample.xseg_mask[..., 0], expected)
    if codec == "PNG":
        assert sample.xseg_mask.min() == 0
        assert sample.xseg_mask.max() == 1


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (np.array([1, 2, 3], dtype=np.uint8), ErrorCode.INVALID_XSEG_MASK),
        (np.zeros((10, 10), dtype=np.uint8), ErrorCode.UNSUPPORTED_METADATA_TYPE),
        (b"arbitrary bytes", ErrorCode.UNSUPPORTED_METADATA_TYPE),
        (np.array([1, 2, 3], dtype=np.float32), ErrorCode.UNSUPPORTED_METADATA_TYPE),
    ],
)
def test_invalid_xseg_is_not_missing(dfl_file, dfl_points, value, code):
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, "xseg_mask": value}))
    assert rejected.value.code == code
    assert rejected.value.rejection.xseg_mask == "invalid"


def test_truncated_xseg_png(dfl_file, dfl_points, mask_buffer):
    buffer = mask_buffer(np.zeros((3, 3), np.uint8))[:-5]
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, "xseg_mask": buffer}))
    assert rejected.value.code == ErrorCode.INVALID_XSEG_MASK


def test_rejects_oversized_xseg_before_pixel_decode(dfl_file, dfl_points, mask_buffer):
    png = bytearray(mask_buffer(np.zeros((3, 3), np.uint8)).tobytes())
    struct.pack_into(">I", png, 16, 60_000)
    struct.pack_into(">I", png, 29, crc32(png[12:29]))
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, "xseg_mask": np.frombuffer(png, np.uint8)}))
    assert rejected.value.code == ErrorCode.RESOURCE_LIMIT
    assert rejected.value.rejection.xseg_mask == "invalid"
