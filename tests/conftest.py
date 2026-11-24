"""Independently written synthetic JPEG and XSeg fixtures."""

from io import BytesIO
import pickle
import struct

import numpy as np
from PIL import Image
import pytest


_DEFAULT = object()


@pytest.fixture
def dfl_points():
    # Include valid coordinates outside the bounds of the synthetic image.
    return (np.arange(136, dtype=np.float32) / 4 - 5).reshape(68, 2)


@pytest.fixture
def dfl_jpeg(dfl_points):
    def build(metadata=_DEFAULT, *, protocol=4, app15=(), size=(17, 13), progressive=False):
        stream = BytesIO()
        Image.new("RGB", size, (89, 143, 177)).save(stream, "JPEG", progressive=progressive)
        jpeg = stream.getvalue()
        if metadata is _DEFAULT:
            metadata = {"landmarks": dfl_points.tolist()}
        if metadata is not None:
            app15 = (*app15, pickle.dumps(metadata, protocol=protocol))
        segments = b"".join(
            b"\xff\xef" + struct.pack(">H", len(payload) + 2) + payload for payload in app15
        )
        assert jpeg[:4] == b"\xff\xd8\xff\xe0"
        after_jfif = 4 + int.from_bytes(jpeg[4:6], "big")
        return jpeg[:after_jfif] + segments + jpeg[after_jfif:]

    return build


@pytest.fixture
def dfl_file(tmp_path, dfl_jpeg):
    def write(*args, **kwargs):
        path = tmp_path / "sample.JPG"
        path.write_bytes(dfl_jpeg(*args, **kwargs))
        return path

    return write


@pytest.fixture
def mask_buffer():
    def encode(pixels, codec="PNG"):
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, codec)
        return np.frombuffer(stream.getvalue(), dtype=np.uint8)

    return encode
