"""JSON and numeric archive integrity for benchmark datasets and predictions."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


def json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True,
    ).encode("ascii")


def digest_json(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(json_bytes((name, array.dtype.str, array.shape)))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _unique_keys(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"Duplicate JSON key {key!r}")
        data[key] = value
    return data


def _reject_constant(value):
    raise ValueError(f"Invalid JSON number {value}")


def read_json(path):
    with path.open(encoding="utf-8") as source:
        value = json.load(source, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_keys(value, expected, description):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{description} requires exactly these fields: {', '.join(sorted(expected))}")


def validate_arrays(arrays, specs, description):
    require_keys(arrays, specs, description)
    for key, (dtype, shape) in specs.items():
        array = arrays[key]
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
            raise ValueError(f"{description} {key!r} must be {np.dtype(dtype).name} with shape {shape}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"{description} {key!r} contains nonfinite values")


def read_arrays(path, specs):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(specs) or set(archive.files) != set(specs):
                raise ValueError(f"Unexpected or duplicate arrays in {path.name}")
            arrays = {key: archive[key] for key in specs}
    except (AttributeError, BadZipFile, EOFError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot load {path.name}: {error}") from error
    validate_arrays(arrays, specs, path.name)
    return arrays


@contextmanager
def publish_directory(destination, *, completion):
    """Stage a flat directory and publish its completion file last, without clobbering."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as stage:
        stage = Path(stage)
        yield stage
        marker = stage / completion
        if not marker.is_file():
            raise RuntimeError(f"Missing completion file {completion}")
        sources = sorted(path for path in stage.iterdir() if path != marker)
        destination.mkdir()
        published = []
        try:
            for source in (*sources, marker):
                target = destination / source.name
                os.link(source, target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink()
            try:
                destination.rmdir()
            except OSError:
                pass
            raise
