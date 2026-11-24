"""DFL recognition follows JPEG markers and preserves original bytes."""

from io import BytesIO
import pickle
from struct import pack

import numpy as np
from PIL import Image
import pytest

from personalab.dfl import ErrorCode, SampleError, read_jpeg


@pytest.mark.parametrize("progressive", [False, True])
def test_reads_original_dimensions_and_pixels(dfl_file, dfl_points, progressive):
    path = dfl_file(progressive=progressive, app15=(b"independent application data",))
    before = path.read_bytes()
    sample = read_jpeg(path)
    assert sample.sample_id == sample.image_ref == "sample.JPG"
    assert sample.image_size == (17, 13)
    np.testing.assert_array_equal(sample.landmarks, dfl_points)
    assert sample.read_image_bytes() == before == path.read_bytes()
    with Image.open(BytesIO(sample.read_image_bytes())) as image:
        assert image.size == sample.image_size


@pytest.mark.parametrize("app15", [(), (b"unrelated APP15",), (pickle.dumps({"other": 1}),)])
def test_missing_dfl_metadata(dfl_file, app15):
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file(None, app15=app15))
    assert rejected.value.code == ErrorCode.MISSING_DFL_METADATA


def test_ignores_app15_inside_another_segment(tmp_path, dfl_jpeg, dfl_points):
    raw = dfl_jpeg(None)
    payload = pickle.dumps({"landmarks": dfl_points})
    nested = b"tag" + b"\xff\xef" + pack(">H", len(payload) + 2) + payload
    path = tmp_path / "nested.jpg"
    path.write_bytes(raw[:2] + b"\xff\xe1" + pack(">H", len(nested) + 2) + nested + raw[2:])
    with pytest.raises(SampleError) as rejected:
        read_jpeg(path)
    assert rejected.value.code == ErrorCode.MISSING_DFL_METADATA


def test_finds_a_real_app15_marker_after_scan_data(tmp_path, dfl_jpeg, dfl_points):
    raw = dfl_jpeg(None, progressive=True)
    payload = pickle.dumps({"landmarks": dfl_points.tolist()}, protocol=4)
    segment = b"\xff\xef" + pack(">H", len(payload) + 2) + payload
    path = tmp_path / "after-scan.jpg"
    path.write_bytes(raw[:-2] + segment + raw[-2:])
    np.testing.assert_array_equal(read_jpeg(path).landmarks, dfl_points)


@pytest.mark.parametrize(
    ("segment", "code"),
    [
        (b"\x01\x10", ErrorCode.INVALID_JPEG),
        (b"\xff\xe1\x00\x01", ErrorCode.INVALID_JPEG),
        (b"\xff\xe1\xff\xff", ErrorCode.INVALID_JPEG),
        (b"\xff\xef\x00\x01", ErrorCode.MALFORMED_APP15),
        (b"\xff\xef\xff\xff", ErrorCode.MALFORMED_APP15),
    ],
)
def test_malformed_markers_and_lengths(tmp_path, dfl_jpeg, segment, code):
    data = dfl_jpeg()
    path = tmp_path / "broken.jpg"
    path.write_bytes(data[:2] + segment + data[2:])
    with pytest.raises(SampleError) as rejected:
        read_jpeg(path)
    assert rejected.value.code == code


@pytest.mark.parametrize("keep_bytes", [1, 10, -1, -20])
def test_truncated_jpeg(tmp_path, dfl_jpeg, keep_bytes):
    path = tmp_path / "truncated.jpeg"
    data = dfl_jpeg(None)
    path.write_bytes(data[:keep_bytes])
    with pytest.raises(SampleError) as rejected:
        read_jpeg(path)
    assert rejected.value.code == ErrorCode.INVALID_JPEG


def test_truncated_pickle_inside_complete_segment(dfl_file, dfl_points):
    payload = pickle.dumps({"landmarks": dfl_points}, protocol=4)[:-1]
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file(None, app15=(payload,)))
    assert rejected.value.code == ErrorCode.MALFORMED_APP15


@pytest.mark.parametrize("same_payload", [True, False])
def test_ambiguous_app15_segments(dfl_file, dfl_points, same_payload):
    metadata = {"landmarks": dfl_points.tolist()}
    other = metadata if same_payload else {"landmarks": (dfl_points + 1).tolist()}
    path = dfl_file(metadata, app15=(pickle.dumps(other, protocol=4),))
    with pytest.raises(SampleError) as rejected:
        read_jpeg(path)
    assert rejected.value.code == ErrorCode.AMBIGUOUS_METADATA
