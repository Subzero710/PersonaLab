"""Bounded JPEG inspection and decoding of DeepFaceLab mask images."""

from dataclasses import dataclass
from io import BytesIO
from struct import unpack_from
from zlib import crc32

import numpy as np
from PIL import Image

from .errors import DFLError, ErrorCode


MAX_JPEG_BYTES = 32 * 1024**2
MAX_JPEG_SIDE = 8_192
MAX_JPEG_PIXELS = 16 * 1024**2
MAX_MASK_SIDE = 4_096
MAX_MASK_PIXELS = 4 * 1024**2
MAX_APP15_SEGMENTS = 16
MAX_JPEG_SEGMENTS = 8_192
MAX_JPEG_SCANS = 128
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class JPEGInfo:
    size: tuple[int, int]
    app15: tuple[bytes, ...]


def _invalid_jpeg(message: str) -> DFLError:
    return DFLError(ErrorCode.INVALID_JPEG, message)


def _check_size(size: tuple[int, int], max_side: int, max_pixels: int, code: ErrorCode):
    width, height = size
    if width < 1 or height < 1:
        raise DFLError(code, "Image dimensions must be positive")
    if max(size) > max_side or width * height > max_pixels:
        raise DFLError(ErrorCode.RESOURCE_LIMIT, "Decoded image dimensions exceed limit")


def scan_jpeg(
    data: bytes, *, max_side: int = MAX_JPEG_SIDE, max_pixels: int = MAX_JPEG_PIXELS,
) -> JPEGInfo:
    """Walk the marker stream, including between progressive scans."""
    if len(data) > MAX_JPEG_BYTES:
        raise DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG file exceeds byte limit")
    if not data.startswith(b"\xff\xd8"):
        raise _invalid_jpeg("Missing JPEG start marker")
    offset = 2
    size = None
    app15 = []
    in_scan = False
    segments = scans = 0
    while offset < len(data):
        if in_scan:
            offset = data.find(b"\xff", offset)
            if offset < 0:
                break
        elif data[offset] != 0xFF:
            raise _invalid_jpeg("Expected a JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset == len(data):
            break
        marker = data[offset]
        offset += 1
        if in_scan and (marker == 0 or 0xD0 <= marker <= 0xD7):
            continue
        in_scan = False
        segments += 1
        if segments > MAX_JPEG_SEGMENTS:
            raise DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG segment count exceeds limit")
        if marker == 0xD9:
            if size is None or not scans or offset != len(data):
                raise _invalid_jpeg("Invalid JPEG end marker or trailing data")
            return JPEGInfo(size, tuple(app15))
        if marker < 0xC0 or 0xD0 <= marker <= 0xD8:
            raise _invalid_jpeg("Unexpected JPEG marker")
        error_code = ErrorCode.MALFORMED_APP15 if marker == 0xEF else ErrorCode.INVALID_JPEG
        if offset + 2 > len(data):
            raise DFLError(error_code, "Missing JPEG segment length")
        length = int.from_bytes(data[offset:offset + 2], "big")
        end = offset + length
        if length < 2 or end > len(data):
            raise DFLError(error_code, "JPEG segment length crosses file boundary")
        start = offset + 2
        if marker in _SOF_MARKERS:
            if marker not in (0xC0, 0xC1, 0xC2) or size is not None or length < 8:
                raise _invalid_jpeg("Unsupported or repeated JPEG frame header")
            precision, height, width, channels = unpack_from(">BHHB", data, start)
            if precision != 8 or not 1 <= channels <= 4 or length != 8 + 3 * channels:
                raise _invalid_jpeg("Invalid JPEG frame layout")
            size = (width, height)
            _check_size(size, max_side, max_pixels, ErrorCode.INVALID_JPEG)
        elif marker == 0xDA:
            if size is None or length < 8:
                raise _invalid_jpeg("Invalid JPEG scan header")
            channels = data[start]
            if not 1 <= channels <= 4 or length != 6 + 2 * channels:
                raise _invalid_jpeg("Invalid JPEG scan layout")
            scans += 1
            if scans > MAX_JPEG_SCANS:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG scan count exceeds limit")
            in_scan = True
        elif marker == 0xEF:
            if len(app15) >= MAX_APP15_SEGMENTS:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG APP15 segment count exceeds limit")
            app15.append(data[start:end])
        offset = end
    raise _invalid_jpeg("Truncated JPEG or missing end marker")


def verify_jpeg_pixels(data: bytes, info: JPEGInfo) -> None:
    """Decode one bounded image to detect damaged pixel data."""
    try:
        with Image.open(BytesIO(data), formats=("JPEG",)) as image:
            if image.size != info.size:
                raise _invalid_jpeg("JPEG frame dimensions disagree with pixel decoder")
            image.load()
    except DFLError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DFLError(ErrorCode.RESOURCE_LIMIT, "Decoded JPEG exceeds limit") from error
    except (OSError, SyntaxError, ValueError) as error:
        raise _invalid_jpeg("JPEG pixel data cannot be decoded") from error


def _mask_png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[8:16] != b"\x00\x00\x00\x0dIHDR":
        raise DFLError(ErrorCode.INVALID_XSEG_MASK, "Invalid XSeg PNG header")
    size = unpack_from(">II", data, 16)
    _check_size(size, MAX_MASK_SIDE, MAX_MASK_PIXELS, ErrorCode.INVALID_XSEG_MASK)
    if data[24:28] != b"\x08\x00\x00\x00" or data[28] not in (0, 1):
        raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Expected an 8-bit grayscale XSeg PNG")

    # Only the chunks emitted for a grayscale mask are needed by the decoder.
    offset = 8
    saw_pixels = False
    while offset + 12 <= len(data):
        length, kind = unpack_from(">I4s", data, offset)
        content_end = offset + 8 + length
        if content_end + 4 > len(data):
            raise DFLError(ErrorCode.INVALID_XSEG_MASK, "Truncated XSeg PNG chunk")
        checksum, = unpack_from(">I", data, content_end)
        if crc32(data[offset + 4:content_end]) != checksum:
            raise DFLError(ErrorCode.INVALID_XSEG_MASK, "Invalid XSeg PNG checksum")
        if kind == b"IEND":
            if saw_pixels and length == 0 and content_end + 4 == len(data):
                return size
            raise DFLError(ErrorCode.INVALID_XSEG_MASK, "Invalid XSeg PNG end chunk")
        if kind == b"IDAT":
            saw_pixels = True
        elif offset != 8 or kind != b"IHDR":
            raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Unsupported XSeg PNG chunk")
        offset = content_end + 4
    raise DFLError(ErrorCode.INVALID_XSEG_MASK, "XSeg PNG is missing its end chunk")


def decode_xseg_mask(value: object) -> np.ndarray:
    if (
        type(value) is not np.ndarray or value.dtype != np.uint8
        or (value.ndim != 1 and (value.ndim != 2 or 1 not in value.shape))
    ):
        raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Expected an encoded uint8 XSeg vector")
    data = value.tobytes()
    if data.startswith(_PNG_SIGNATURE):
        size = _mask_png_size(data)
        codec = "PNG"
    elif data.startswith(b"\xff\xd8"):
        try:
            size = scan_jpeg(data, max_side=MAX_MASK_SIDE, max_pixels=MAX_MASK_PIXELS).size
        except DFLError as error:
            if error.code == ErrorCode.RESOURCE_LIMIT:
                raise
            raise DFLError(ErrorCode.INVALID_XSEG_MASK, "Malformed XSeg JPEG") from error
        codec = "JPEG"
    else:
        raise DFLError(ErrorCode.INVALID_XSEG_MASK, "XSeg buffer lacks a PNG or JPEG signature")
    try:
        with Image.open(BytesIO(data), formats=(codec,)) as image:
            if image.mode != "L":
                raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Expected grayscale XSeg pixels")
            if image.size != size:
                raise DFLError(ErrorCode.INVALID_XSEG_MASK, "XSeg mask dimensions disagree")
            result = np.asarray(image, dtype=np.float32)[..., None] / np.float32(255)
    except DFLError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DFLError(ErrorCode.RESOURCE_LIMIT, "Decoded XSeg mask exceeds limit") from error
    except (OSError, SyntaxError, ValueError) as error:
        raise DFLError(ErrorCode.INVALID_XSEG_MASK, "XSeg pixels cannot be decoded") from error
    result.setflags(write=False)
    return result
