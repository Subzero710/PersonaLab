"""Stream DFL samples and collect validation statistics."""

from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import stat
from typing import BinaryIO, Literal

import numpy as np

from .errors import DatasetError, DFLError, ErrorCode, ValidationIssue
from .jpeg import MAX_JPEG_BYTES, scan_jpeg, verify_jpeg_pixels
from .metadata import OPTIONAL_FIELDS, find_metadata, inspect_metadata
from .zipio import MAX_ZIP_EXPANDED_BYTES, check_zip_member, open_zip, read_zip_member, safe_zip_name


_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
_MAX_DIRECTORY_ENTRIES = 20_000
_MAX_DIRECTORY_DEPTH = 64
_MAX_REFERENCE_BYTES = 1_024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_InputKind = Literal["file", "directory", "zip"]


class FieldStatus(StrEnum):
    UNKNOWN = "unknown"
    MISSING = "missing"
    INVALID = "invalid"
    VALID = "valid"


@dataclass(frozen=True)
class _ImageSource:
    root: Path
    kind: _InputKind
    reference: str

    def read_bytes(self) -> bytes:
        if self.kind == "file":
            return _read_file(self.root)
        if self.kind == "directory":
            with ExitStack() as stack:
                directory = stack.enter_context(_directory_fd(self.root))
                parts = self.reference.split("/")
                for part in parts[:-1]:
                    directory = stack.enter_context(_directory_fd(part, parent=directory))
                return _read_file(parts[-1], directory=directory)
        with _regular_file(self.root) as stream, open_zip(stream) as archive:
            entries = [info for info in archive.infolist() if info.orig_filename == self.reference]
            if len(entries) != 1:
                raise DFLError(ErrorCode.DATASET_CHANGED, "Original ZIP entry is missing or ambiguous")
            info = entries[0]
            check_zip_member(info)
            return read_zip_member(stream, archive, info)


@dataclass(frozen=True, eq=False, kw_only=True)
class DFLFaceSample:
    """Validated aligned JPEG metadata and a reference to its original image."""

    sample_id: str
    image_ref: str
    image_size: tuple[int, int]
    landmarks: np.ndarray
    face_type: str | None = None
    source_filename: str | None = None
    source_rect: np.ndarray | None = None
    source_landmarks: np.ndarray | None = None
    image_to_face_mat: np.ndarray | None = None
    xseg_mask: np.ndarray | None = None
    additional_metadata: dict = field(default_factory=dict)
    _source: _ImageSource = field(repr=False)
    _digest: bytes = field(repr=False)

    def read_image_bytes(self) -> bytes:
        """Reopen the input and return the original JPEG bytes, including metadata."""
        try:
            data = self._source.read_bytes()
        except (DFLError, OSError) as error:
            raise DFLError(ErrorCode.DATASET_CHANGED, "Original JPEG is no longer accessible") from error
        if sha256(data).digest() != self._digest:
            raise DFLError(ErrorCode.DATASET_CHANGED, "Original JPEG bytes have changed")
        return data


@dataclass(frozen=True)
class SampleRejection:
    """One discovered image that could not be imported."""

    sample_id: str
    image_ref: str | None
    issues: tuple[ValidationIssue, ...]
    image_size: tuple[int, int] | None = None
    landmarks: FieldStatus = FieldStatus.UNKNOWN
    xseg_mask: FieldStatus = FieldStatus.UNKNOWN
    missing_optional_fields: tuple[str, ...] = ()


class SampleError(DFLError):
    """The JPEG requested through read_jpeg was rejected."""

    def __init__(self, rejection: SampleRejection):
        first_issue = rejection.issues[0]
        super().__init__(first_issue.code, first_issue.message)
        self.rejection = rejection


_SampleResult = DFLFaceSample | SampleRejection


@contextmanager
def _directory_fd(path: str | Path, *, parent: int | None = None) -> Iterator[int]:
    fd = os.open(path, _DIRECTORY_FLAGS, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _regular_file(path: str | Path, *, directory: int | None = None) -> Iterator[BinaryIO]:
    mode = os.stat(path, dir_fd=directory, follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise DFLError(ErrorCode.UNSAFE_PATH, "Expected a regular file; symbolic links are refused")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DFLError(ErrorCode.UNSAFE_PATH, "Opened entry is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(fd)


def _read_file(path: str | Path, *, directory: int | None = None) -> bytes:
    try:
        with _regular_file(path, directory=directory) as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > MAX_JPEG_BYTES:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG file exceeds byte limit")
            data = stream.read(size + 1)
            if len(data) > MAX_JPEG_BYTES:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG file exceeds byte limit")
            if len(data) != size:
                raise DFLError(ErrorCode.DATASET_CHANGED, "JPEG file size changed during reading")
            return data
    except OSError as error:
        raise DFLError(ErrorCode.JPEG_UNREADABLE, "JPEG file could not be read") from error


def _kind(path: Path) -> _InputKind:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise DatasetError(ErrorCode.MISSING_INPUT, "Input does not exist") from error
    except ValueError as error:
        raise DatasetError(ErrorCode.UNSUPPORTED_INPUT, "Invalid input path") from error
    except OSError as error:
        raise DatasetError(ErrorCode.INPUT_UNREADABLE, "Input could not be inspected") from error
    if stat.S_ISLNK(mode):
        raise DatasetError(ErrorCode.UNSAFE_PATH, "The input is a symbolic link")
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        if path.suffix.lower() in _JPEG_EXTENSIONS:
            return "file"
        if path.suffix.lower() == ".zip":
            return "zip"
    raise DatasetError(ErrorCode.UNSUPPORTED_INPUT, "Expected a JPEG, directory or ZIP")


def _reject(sample_id: str, reference: str | None, error: DFLError, size=None) -> SampleRejection:
    return SampleRejection(sample_id, reference, (ValidationIssue(error.code, str(error)),), size)


def _inspect(source: _ImageSource, sample_id: str, data: bytes) -> _SampleResult:
    size = None
    try:
        info = scan_jpeg(data)
        size = info.size
        verify_jpeg_pixels(data, info)
        inspection = inspect_metadata(find_metadata(info.app15))
    except DFLError as error:
        return _reject(sample_id, source.reference, error, size)
    if inspection.issues:
        invalid = {issue.field for issue in inspection.issues}
        if any(issue.code == ErrorCode.MISSING_LANDMARKS for issue in inspection.issues):
            landmarks = FieldStatus.MISSING
        else:
            landmarks = FieldStatus.INVALID if "landmarks" in invalid else FieldStatus.VALID
        if "xseg_mask" in inspection.missing_optional:
            xseg = FieldStatus.MISSING
        else:
            xseg = FieldStatus.INVALID if "xseg_mask" in invalid else FieldStatus.VALID
        return SampleRejection(
            sample_id, source.reference, inspection.issues, size,
            landmarks, xseg, inspection.missing_optional,
        )
    return DFLFaceSample(
        sample_id=sample_id, image_ref=source.reference, image_size=size,
        additional_metadata=inspection.additional, _source=source,
        _digest=sha256(data).digest(), **inspection.fields,
    )


def _inspect_file(source: _ImageSource, path: str | Path, *, directory=None) -> _SampleResult:
    try:
        data = _read_file(path, directory=directory)
    except DFLError as error:
        return _reject(source.reference, source.reference, error)
    return _inspect(source, source.reference, data)


def read_jpeg(path: str | Path) -> DFLFaceSample:
    """Read one JPEG, raising DatasetError or SampleError on failure."""
    root = Path(path).absolute()
    if _kind(root) != "file":
        raise DatasetError(ErrorCode.UNSUPPORTED_INPUT, "Expected a single JPEG file")
    result = _inspect_file(_ImageSource(root, "file", root.name), root)
    if isinstance(result, SampleRejection):
        raise SampleError(result)
    return result


def _iter_directory(root: Path) -> Iterator[_SampleResult]:
    entry_count = 0

    def walk(directory: int, parents: tuple[str, ...]) -> Iterator[_SampleResult]:
        nonlocal entry_count
        entries = []
        with os.scandir(directory) as listing:
            for entry in listing:
                entry_count += 1
                if entry_count > _MAX_DIRECTORY_ENTRIES:
                    raise DatasetError(ErrorCode.RESOURCE_LIMIT, "Directory entry count exceeds limit")
                entries.append(entry)
        for entry in sorted(entries, key=lambda item: item.name):
            parts = (*parents, entry.name)
            reference = "/".join(parts)
            if entry.is_dir(follow_symlinks=False):
                if len(parts) >= _MAX_DIRECTORY_DEPTH or len(os.fsencode(reference)) > _MAX_REFERENCE_BYTES:
                    raise DatasetError(ErrorCode.RESOURCE_LIMIT, "Dataset directory depth or path exceeds limit")
                with _directory_fd(entry.name, parent=directory) as child:
                    yield from walk(child, parts)
            elif Path(entry.name).suffix.lower() in _JPEG_EXTENSIONS:
                source = _ImageSource(root, "directory", reference)
                if len(os.fsencode(reference)) > _MAX_REFERENCE_BYTES:
                    error = DFLError(ErrorCode.RESOURCE_LIMIT, "JPEG relative path exceeds limit")
                    yield _reject(reference, reference, error)
                else:
                    yield _inspect_file(source, entry.name, directory=directory)

    with _directory_fd(root) as directory:
        yield from walk(directory, ())


def _iter_zip(root: Path) -> Iterator[_SampleResult]:
    expanded_bytes = 0
    with _regular_file(root) as stream, open_zip(stream) as archive:
        candidates = [
            (index, info) for index, info in enumerate(archive.infolist())
            if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() in _JPEG_EXTENSIONS
        ]
        duplicates = Counter(info.orig_filename for _, info in candidates)
        for index, info in sorted(candidates, key=lambda item: (item[1].orig_filename, item[0])):
            reference = info.orig_filename
            entry_id = f"zip-entry:{index}"
            if not safe_zip_name(reference):
                error = DFLError(ErrorCode.UNSAFE_PATH, "Unsafe ZIP member path")
                yield _reject(entry_id, None, error)
                continue
            if duplicates[reference] > 1:
                error = DFLError(ErrorCode.DUPLICATE_ZIP_ENTRY, "Duplicate ZIP member name")
                yield _reject(entry_id, reference, error)
                continue
            sample_id = f"zip:{reference}"
            try:
                check_zip_member(info)
                expanded_bytes += info.file_size
                if expanded_bytes > MAX_ZIP_EXPANDED_BYTES:
                    raise DatasetError(ErrorCode.RESOURCE_LIMIT, "ZIP JPEG expansion budget exceeded")
                data = read_zip_member(stream, archive, info)
            except DatasetError:
                raise
            except DFLError as error:
                yield _reject(sample_id, reference, error)
                continue
            result = _inspect(_ImageSource(root, "zip", reference), sample_id, data)
            del data
            yield result


def iter_dataset(path: str | Path) -> Iterator[_SampleResult]:
    """Yield one accepted sample or rejection per discovered JPEG."""
    root = Path(path).absolute()
    kind = _kind(root)
    try:
        if kind == "file":
            yield _inspect_file(_ImageSource(root, kind, root.name), root)
        elif kind == "directory":
            yield from _iter_directory(root)
        else:
            yield from _iter_zip(root)
    except DatasetError:
        raise
    except OSError as error:
        raise DatasetError(ErrorCode.INPUT_UNREADABLE, "Dataset could not be scanned") from error
    except DFLError as error:
        raise DatasetError(error.code, str(error)) from error


@dataclass
class ValidationReport:
    discovered_samples: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0
    resolution_distribution: dict[str, int] = field(default_factory=dict)
    unknown_resolution: int = 0
    landmarks: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FieldStatus, 0))
    xseg_masks: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FieldStatus, 0))
    missing_optional_metadata: dict[str, int] = field(default_factory=lambda: dict.fromkeys(OPTIONAL_FIELDS, 0))
    inconsistent_metadata: int = 0
    unsupported_metadata: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejections: list[SampleRejection] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-compatible summary with relative image references."""
        return asdict(self)


def validate_dataset(path: str | Path) -> ValidationReport:
    """Scan the dataset and retain counters and rejections only."""
    report = ValidationReport()
    rejection_counts = Counter()
    resolutions = Counter()
    for result in iter_dataset(path):
        report.discovered_samples += 1
        if result.image_size is None:
            report.unknown_resolution += 1
        else:
            width, height = result.image_size
            resolutions[f"{width}x{height}"] += 1
        if isinstance(result, DFLFaceSample):
            report.accepted_samples += 1
            landmarks = FieldStatus.VALID
            xseg = FieldStatus.MISSING if result.xseg_mask is None else FieldStatus.VALID
            missing = (
                name for name in OPTIONAL_FIELDS
                if getattr(result, name, result.additional_metadata.get(name)) is None
            )
        else:
            report.rejected_samples += 1
            report.rejections.append(result)
            codes = {issue.code for issue in result.issues}
            rejection_counts.update(codes)
            report.inconsistent_metadata += bool(codes & {
                ErrorCode.AMBIGUOUS_METADATA, ErrorCode.INVALID_ALIGNMENT_METADATA,
                ErrorCode.INVALID_TRANSFORM,
            }) or any(issue.field == "source_landmarks" for issue in result.issues)
            report.unsupported_metadata += ErrorCode.UNSUPPORTED_METADATA_TYPE in codes
            landmarks, xseg = result.landmarks, result.xseg_mask
            missing = result.missing_optional_fields
        report.landmarks[landmarks] += 1
        report.xseg_masks[xseg] += 1
        for name in missing:
            report.missing_optional_metadata[name] += 1
    report.rejection_counts = dict(rejection_counts)
    report.resolution_distribution = dict(resolutions)
    return report
