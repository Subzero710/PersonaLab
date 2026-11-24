"""Failure categories shared by the DeepFaceLab readers."""

from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    MISSING_INPUT = "missing_input"
    UNSUPPORTED_INPUT = "unsupported_input"
    INPUT_UNREADABLE = "input_unreadable"
    JPEG_UNREADABLE = "jpeg_unreadable"
    UNSAFE_PATH = "unsafe_path"
    INVALID_JPEG = "invalid_jpeg"
    MISSING_DFL_METADATA = "missing_dfl_metadata"
    MALFORMED_APP15 = "malformed_app15"
    AMBIGUOUS_METADATA = "ambiguous_metadata"
    UNSAFE_SERIALIZATION = "unsafe_serialization"
    UNSUPPORTED_METADATA_TYPE = "unsupported_metadata_type"
    MISSING_LANDMARKS = "missing_landmarks"
    INVALID_LANDMARK_SHAPE = "invalid_landmark_shape"
    INVALID_LANDMARK_TYPE = "invalid_landmark_type"
    NON_FINITE_LANDMARKS = "non_finite_landmarks"
    INVALID_ALIGNMENT_METADATA = "invalid_alignment_metadata"
    INVALID_TRANSFORM = "invalid_transform"
    INVALID_XSEG_MASK = "invalid_xseg_mask"
    CORRUPT_ZIP = "corrupt_zip"
    CORRUPT_ZIP_ENTRY = "corrupt_zip_entry"
    UNSUPPORTED_ZIP_ENTRY = "unsupported_zip_entry"
    ENCRYPTED_ZIP_ENTRY = "encrypted_zip_entry"
    DUPLICATE_ZIP_ENTRY = "duplicate_zip_entry"
    RESOURCE_LIMIT = "resource_limit"
    DATASET_CHANGED = "dataset_changed"


class DFLError(ValueError):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code


class DatasetError(DFLError):
    """The input could not be opened or scanned as a whole."""


@dataclass(frozen=True)
class ValidationIssue:
    code: ErrorCode
    message: str
    field: str | None = None
