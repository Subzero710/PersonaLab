"""Identify DFL metadata and validate its documented alignment fields."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .errors import DFLError, ErrorCode, ValidationIssue
from .jpeg import decode_xseg_mask
from .serialization import decode_metadata


def _is_number(value: object) -> bool:
    return type(value) in (int, float) or isinstance(value, (np.integer, np.floating))


def _numeric_array(
    value, shape: tuple[int | None, ...], *,
    shape_error: ErrorCode, type_error: ErrorCode, finite_error: ErrorCode,
) -> np.ndarray:
    original = []
    if type(value) is np.ndarray:
        array = value
    elif type(value) in (list, tuple):
        if len(shape) == 2:
            if any(type(row) not in (list, tuple) or len(row) != shape[1] for row in value):
                raise DFLError(shape_error, "Invalid coordinate row shape")
            original = [item for row in value for item in row]
        else:
            original = list(value)
        if any(not _is_number(item) for item in original):
            raise DFLError(type_error, "Expected real numeric coordinates")
        try:
            array = np.asarray(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise DFLError(type_error, "Numeric sequence cannot be represented by NumPy") from error
    else:
        raise DFLError(type_error, "Expected a numeric array or sequence")

    if array.ndim != len(shape) or any(
        expected is not None and size != expected
        for size, expected in zip(array.shape, shape)
    ):
        raise DFLError(shape_error, "Invalid coordinate array shape")
    if array.dtype.kind not in "iuf":
        raise DFLError(type_error, "Expected real numeric coordinates")
    if not np.isfinite(array).all():
        raise DFLError(finite_error, "Coordinates must be finite")
    for converted, requested in zip(array.flat, original):
        if converted.item() != (requested.item() if isinstance(requested, np.generic) else requested):
            raise DFLError(type_error, "Numeric sequence cannot be represented without precision loss")
    array.setflags(write=False)
    return array


def _points(value):
    return _numeric_array(
        value, (68, 2),
        shape_error=ErrorCode.INVALID_LANDMARK_SHAPE,
        type_error=ErrorCode.INVALID_LANDMARK_TYPE,
        finite_error=ErrorCode.NON_FINITE_LANDMARKS,
    )


def _landmarks(value):
    if value is None:
        raise DFLError(ErrorCode.MISSING_LANDMARKS, "Expected 68 aligned landmarks")
    return _points(value)


def _text(value):
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise DFLError(ErrorCode.INVALID_ALIGNMENT_METADATA, "Expected nonempty text")
    return value


def _face_type(value):
    value = _text(value)
    if value.lower() == "mark_only":
        raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "mark_only stores an unaligned image")
    return value


def _alignment_array(value, shape, code=ErrorCode.INVALID_ALIGNMENT_METADATA):
    return _numeric_array(value, shape, shape_error=code, type_error=code, finite_error=code)


def _source_rect(value):
    rect = _alignment_array(value, (4,))
    left, top, right, bottom = rect
    if left >= right or top >= bottom:
        raise DFLError(ErrorCode.INVALID_ALIGNMENT_METADATA, "Source rectangle edges are inverted or equal")
    return rect


def _transform(value):
    return _alignment_array(value, (2, 3), ErrorCode.INVALID_TRANSFORM)


def _eyebrows(value):
    if not _is_number(value) or not isfinite(value):
        raise DFLError(ErrorCode.INVALID_ALIGNMENT_METADATA, "Expected a finite eyebrow modifier")
    return value


def _polygons(value):
    error = DFLError(ErrorCode.INVALID_ALIGNMENT_METADATA, "Invalid segmentation polygon data")
    if type(value) is dict and set(value) == {"polys"} and type(value["polys"]) is list:
        entries = value["polys"]
        if any(type(entry) is not dict or set(entry) != {"type", "pts"} for entry in entries):
            raise error
        pairs = [(entry["type"], entry["pts"]) for entry in entries]
    elif type(value) is list:
        if any(type(entry) not in (list, tuple) or len(entry) != 2 for entry in value):
            raise error
        pairs = value
    else:
        raise error
    for kind, points in pairs:
        if type(kind) is not int or kind not in (0, 1):
            raise error
        _alignment_array(points, (None, 2))
    return value


_FIELDS = {
    "landmarks": _landmarks,
    "face_type": _face_type,
    "source_filename": _text,
    "source_rect": _source_rect,
    "source_landmarks": _points,
    "image_to_face_mat": _transform,
    "xseg_mask": decode_xseg_mask,
}
_EXTRA_FIELDS = {"eyebrows_expand_mod": _eyebrows, "seg_ie_polys": _polygons}
_DFL_KEYS = _FIELDS.keys() | _EXTRA_FIELDS.keys()
OPTIONAL_FIELDS = tuple(key for key in (*_FIELDS, *_EXTRA_FIELDS) if key != "landmarks")


def find_metadata(app15: tuple[bytes, ...]) -> dict:
    """Recognize DFL's unprefixed pickle dictionary by its documented keys."""
    found = None
    for payload in app15:
        if payload.startswith((b"(dp", b"}q")):
            raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Legacy text or protocol 1 pickle")
        if not payload.startswith(b"\x80"):
            continue
        data = decode_metadata(payload)
        if _DFL_KEYS.isdisjoint(data):
            continue
        if found is not None:
            raise DFLError(ErrorCode.AMBIGUOUS_METADATA, "Multiple DeepFaceLab APP15 segments")
        found = data
    if found is None:
        raise DFLError(ErrorCode.MISSING_DFL_METADATA, "No DeepFaceLab APP15 dictionary")
    return found


@dataclass
class MetadataInspection:
    fields: dict
    additional: dict
    issues: tuple[ValidationIssue, ...]
    missing_optional: tuple[str, ...]


def inspect_metadata(metadata: dict) -> MetadataInspection:
    fields = {}
    additional = {key: value for key, value in metadata.items() if key not in _DFL_KEYS}
    issues = []
    missing = []
    for readers, destination in ((_FIELDS, fields), (_EXTRA_FIELDS, additional)):
        for field, validate in readers.items():
            value = metadata.get(field)
            if value is None and field != "landmarks":
                missing.append(field)
                continue
            try:
                destination[field] = validate(value)
            except DFLError as error:
                issues.append(ValidationIssue(error.code, str(error), field))
    return MetadataInspection(fields, additional, tuple(issues), tuple(missing))
