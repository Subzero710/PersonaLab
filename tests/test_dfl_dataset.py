"""Directory and ZIP integration, resource cleanup, and streaming measurements."""

import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo
from zlib import crc32

import numpy as np
import pytest

from personalab.dfl import (
    DFLFaceSample, DatasetError, DFLError, ErrorCode, SampleRejection,
    iter_dataset, validate_dataset,
)


def test_single_input_iterator(dfl_file):
    path = dfl_file()
    sample, = iter_dataset(path)
    assert isinstance(sample, DFLFaceSample)
    assert sample.image_ref == path.name
    assert sample.read_image_bytes() == path.read_bytes()


def test_directory_order_and_rejection_counts(tmp_path, dfl_jpeg, dfl_points, mask_buffer):
    for name in ("z", "a"):
        (tmp_path / name).mkdir()
    good_with_mask = dfl_jpeg({
        "landmarks": dfl_points, "xseg_mask": mask_buffer(np.full((4, 4), 71, dtype=np.uint8)),
    })
    invalid = dfl_jpeg({
        "landmarks": np.full((68, 2), np.nan), "image_to_face_mat": [[0]],
        "xseg_mask": np.arange(4, dtype=np.uint8), "source_filename": "private/source.png",
    })
    files = {
        "z/same.jpeg": good_with_mask,
        "middle.jpg": invalid,
        "bad.jpg": b"JPEG name, invalid bytes",
        "a/same.JPG": dfl_jpeg(),
        "ignored.png": b"ignored data",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    results = list(iter_dataset(tmp_path))
    assert [item.sample_id for item in results] == [
        "a/same.JPG", "bad.jpg", "middle.jpg", "z/same.jpeg",
    ]
    first, bad_jpeg, middle, last = results
    assert isinstance(first, DFLFaceSample)
    assert bad_jpeg.issues[0].code == ErrorCode.INVALID_JPEG
    assert {issue.code for issue in middle.issues} == {
        ErrorCode.NON_FINITE_LANDMARKS, ErrorCode.INVALID_TRANSFORM, ErrorCode.INVALID_XSEG_MASK,
    }
    assert isinstance(last, DFLFaceSample)
    assert first.read_image_bytes() == files[first.image_ref]
    assert last.read_image_bytes() == files[last.image_ref]

    serialized = json.dumps(validate_dataset(tmp_path).to_dict())
    report = json.loads(serialized)
    assert report["discovered_samples"] == 4
    assert report["accepted_samples"] == report["rejected_samples"] == 2
    assert report["resolution_distribution"] == {"17x13": 3}
    assert report["unknown_resolution"] == 1
    assert report["landmarks"] == {"valid": 2, "invalid": 1, "missing": 0, "unknown": 1}
    assert report["xseg_masks"] == {"valid": 1, "invalid": 1, "missing": 1, "unknown": 1}
    assert report["missing_optional_metadata"]["image_to_face_mat"] == 2
    assert report["inconsistent_metadata"] == 1
    assert report["unsupported_metadata"] == 0
    assert report["rejection_counts"] == {
        "invalid_jpeg": 1, "non_finite_landmarks": 1, "invalid_transform": 1, "invalid_xseg_mask": 1,
    }
    assert [item["image_ref"] for item in report["rejections"]] == ["bad.jpg", "middle.jpg"]
    assert "private/source.png" not in serialized
    assert str(tmp_path) not in serialized
    for name, data in files.items():
        assert (tmp_path / name).read_bytes() == data


def test_directory_does_not_follow_symlinks(tmp_path, dfl_jpeg):
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.jpg").write_bytes(dfl_jpeg())
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "outside").symlink_to(external, target_is_directory=True)
    (dataset / "outside.jpg").symlink_to(external / "secret.jpg")
    (dataset / "valid.jpeg").write_bytes(dfl_jpeg())
    rejected, accepted = iter_dataset(dataset)
    assert rejected.image_ref == "outside.jpg"
    assert rejected.issues[0].code == ErrorCode.UNSAFE_PATH
    assert accepted.image_ref == "valid.jpeg"
    with pytest.raises(DatasetError) as root_error:
        list(iter_dataset(dataset / "outside"))
    assert root_error.value.code == ErrorCode.UNSAFE_PATH


def test_archive_order_duplicates_and_nested_members(tmp_path, dfl_jpeg):
    path = tmp_path / "mixed.ZIP"
    data = dfl_jpeg()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("z/same.JPG", data)
        archive.writestr("z/", b"")
        archive.writestr("no-metadata.jpg", dfl_jpeg(None))
        archive.writestr("a/nested/same.JPG", data)
        archive.writestr("duplicate.jpg", data)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("duplicate.jpg", data)
        archive.writestr("ignored.txt", b"not a JPEG")
    before = path.read_bytes()
    results = list(iter_dataset(path))
    assert [item.image_ref for item in results] == [
        "a/nested/same.JPG", "duplicate.jpg", "duplicate.jpg", "no-metadata.jpg", "z/same.JPG",
    ]
    assert len({item.sample_id for item in results}) == 5
    assert [item.issues[0].code for item in results[1:4]] == [
        ErrorCode.DUPLICATE_ZIP_ENTRY, ErrorCode.DUPLICATE_ZIP_ENTRY, ErrorCode.MISSING_DFL_METADATA,
    ]
    assert results[0].read_image_bytes() == data == results[4].read_image_bytes()
    summary = validate_dataset(path)
    assert (summary.discovered_samples, summary.accepted_samples, summary.rejected_samples) == (5, 2, 3)
    assert path.read_bytes() == before


def test_archive_paths_and_symlinks_are_rejected(tmp_path, dfl_jpeg):
    path = tmp_path / "paths.zip"
    link = ZipInfo("link.jpg")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    names = ["../outside.jpg", "/outside.jpg", "C:\\outside.jpg", "nested//image.jpg"]
    with ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, dfl_jpeg())
        archive.writestr(link, b"../outside.jpg")
        archive.writestr("safe/image.jpg", dfl_jpeg())
    results = list(iter_dataset(path))
    rejected = [item for item in results if isinstance(item, SampleRejection)]
    assert len(rejected) == 5
    assert {item.issues[0].code for item in rejected} == {ErrorCode.UNSAFE_PATH}
    assert sum(item.image_ref is None for item in rejected) == len(names)
    assert {item.image_ref for item in results if isinstance(item, DFLFaceSample)} == {"safe/image.jpg"}
    assert list(tmp_path.iterdir()) == [path]


def test_archive_expansion_ratio_is_bounded(tmp_path, dfl_jpeg):
    path = tmp_path / "expansion.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ratio.jpg", bytes(512_000))
        archive.writestr("valid.jpg", dfl_jpeg())
    rejected, accepted = iter_dataset(path)
    assert rejected.issues[0].code == ErrorCode.RESOURCE_LIMIT
    assert isinstance(accepted, DFLFaceSample)


def test_archive_corruption_and_unsupported_entries(tmp_path, dfl_jpeg):
    path = tmp_path / "damaged.zip"
    data = dfl_jpeg()
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("broken.jpg", data)
        archive.writestr("locked.jpg", data)
        archive.writestr("unsupported.jpg", data, compress_type=ZIP_BZIP2)
        archive.writestr("valid.jpg", data)
        local_locked = archive.getinfo("locked.jpg").header_offset
    raw = bytearray(path.read_bytes())
    raw[30 + len("broken.jpg")] ^= 1  # First byte of the first member; leave its CRC unchanged.
    central_start, = struct.unpack_from("<I", raw, len(raw) - 6)
    central_locked = central_start + 46 + len("broken.jpg")
    raw[local_locked + 6] |= 1
    raw[central_locked + 8] |= 1
    path.write_bytes(raw)

    broken, encrypted, unsupported, accepted = iter_dataset(path)
    assert broken.issues[0].code == ErrorCode.CORRUPT_ZIP_ENTRY
    assert encrypted.issues[0].code == ErrorCode.ENCRYPTED_ZIP_ENTRY
    assert unsupported.issues[0].code == ErrorCode.UNSUPPORTED_ZIP_ENTRY
    assert isinstance(accepted, DFLFaceSample)
    assert path.read_bytes() == raw


def test_archive_member_actual_expansion_must_match_declared_length(tmp_path, dfl_jpeg):
    path = tmp_path / "wrong-size.zip"
    jpeg = dfl_jpeg()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("image.jpg", jpeg + b"extra fixture bytes")
    raw = bytearray(path.read_bytes())
    central_start, = struct.unpack_from("<I", raw, len(raw) - 6)
    struct.pack_into("<I", raw, 14, crc32(jpeg))
    struct.pack_into("<I", raw, 22, len(jpeg))
    struct.pack_into("<I", raw, central_start + 16, crc32(jpeg))
    struct.pack_into("<I", raw, central_start + 24, len(jpeg))
    path.write_bytes(raw)
    rejected, = iter_dataset(path)
    assert rejected.issues[0].code == ErrorCode.CORRUPT_ZIP_ENTRY


def test_dataset_failures_are_distinct_from_sample_rejections(tmp_path):
    with pytest.raises(DatasetError) as missing:
        validate_dataset(tmp_path / "missing")
    assert missing.value.code == ErrorCode.MISSING_INPUT
    unsupported = tmp_path / "images.tar"
    unsupported.touch()
    with pytest.raises(DatasetError) as error:
        validate_dataset(unsupported)
    assert error.value.code == ErrorCode.UNSUPPORTED_INPUT
    zip_path = tmp_path / "not-a-zip.zip"
    zip_path.write_bytes(b"ZIP filename without a ZIP directory")
    with pytest.raises(DatasetError) as corrupted:
        validate_dataset(zip_path)
    assert corrupted.value.code == ErrorCode.CORRUPT_ZIP


def test_changed_jpeg_is_not_returned_by_sample(dfl_file):
    path = dfl_file()
    sample, = iter_dataset(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(DFLError) as changed:
        sample.read_image_bytes()
    assert changed.value.code == ErrorCode.DATASET_CHANGED


def _open_dataset_fds(path):
    targets = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            targets.append(os.readlink(fd))
        except FileNotFoundError:
            continue
    return [target for target in targets if target == str(path) or target.startswith(f"{path}/")]


@pytest.mark.parametrize("container", ["directory", "zip"])
def test_closing_iterator_releases_dataset_handles(tmp_path, dfl_jpeg, container):
    data = dfl_jpeg()
    if container == "directory":
        path = tmp_path / "input"
        path.mkdir()
        (path / "image.jpg").write_bytes(data)
    else:
        path = tmp_path / "input.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("image.jpg", data)
    assert not _open_dataset_fds(path)
    iterator = iter_dataset(path)
    try:
        sample = next(iterator)
        assert _open_dataset_fds(path)
    finally:
        iterator.close()
    assert not _open_dataset_fds(path)
    assert sample.read_image_bytes() == data
    assert not _open_dataset_fds(path)


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "personalab.dfl_import", *map(str, args)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False,
    )


def test_cli_json_and_no_clobber(tmp_path, dfl_jpeg):
    archive_path = tmp_path / "cli.zip"
    before = dfl_jpeg()
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("valid.jpg", before)
        archive.writestr("invalid.jpg", b"invalid")
    output = tmp_path / "report.json"
    cli = _run_cli("--input", archive_path, "--output", output)
    assert cli.returncode == 1, cli.stderr
    payload = output.read_bytes()
    assert json.loads(payload)["accepted_samples"] == 1
    assert json.loads(payload)["rejections"][0]["image_ref"] == "invalid.jpg"
    replay = _run_cli("--input", archive_path, "--output", output)
    assert replay.returncode == 2
    assert output.read_bytes() == payload
    cli = _run_cli("--input", tmp_path / "missing.zip")
    assert cli.returncode == 2
    assert json.loads(cli.stdout)["dataset_error"]["code"] == "missing_input"


def test_cli_directory_stdout_and_protects_dataset(tmp_path, dfl_jpeg):
    (tmp_path / "valid.jpg").write_bytes(dfl_jpeg())
    cli = _run_cli("--input", tmp_path)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["accepted_samples"] == 1
    inside = tmp_path / "report.json"
    blocked = _run_cli("--input", tmp_path, "--output", inside)
    assert blocked.returncode == 2
    assert not inside.exists()


_MEMORY_PROBE = """
import json
from pathlib import Path
import sys
import time
import tracemalloc

from personalab.dfl import DFLFaceSample, iter_dataset

def peak_rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise RuntimeError("Linux VmHWM is unavailable")

baseline = peak_rss_kib()
tracemalloc.start()
start = time.perf_counter()
count = 0
for sample in iter_dataset(sys.argv[1]):
    assert isinstance(sample, DFLFaceSample), sample
    assert sample.image_size == (512, 512)
    assert sample.xseg_mask.shape == (256, 256, 1)
    count += 1
duration = time.perf_counter() - start
_, traced_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({
    "samples": count, "elapsed_seconds": duration, "peak_traced_bytes": traced_peak,
    "baseline_rss_kib": baseline, "peak_rss_kib": peak_rss_kib(),
}))
"""


@pytest.mark.parametrize("container", ["directory", "zip"])
def test_streaming_memory(tmp_path, dfl_jpeg, dfl_points, mask_buffer, container):
    total = 600
    buffer = mask_buffer(np.full((256, 256), 121, dtype=np.uint8))
    data = dfl_jpeg({"landmarks": dfl_points, "xseg_mask": buffer}, size=(512, 512))
    if container == "directory":
        path = tmp_path / "large"
        path.mkdir()
        for index in range(total):
            (path / f"{index:04d}.jpg").write_bytes(data)
    else:
        path = tmp_path / "large.zip"
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for index in range(total):
                archive.writestr(f"{index:04d}.jpg", data)
    worker = subprocess.run(
        [sys.executable, "-c", _MEMORY_PROBE, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=True, timeout=180,
    )
    metrics = json.loads(worker.stdout)
    print(f"{container} {json.dumps(metrics, sort_keys=True)}")
    assert metrics["samples"] == total
    assert metrics["peak_rss_kib"] - metrics["baseline_rss_kib"] < 48 * 1024
    assert metrics["peak_traced_bytes"] < 24 * 1024**2
