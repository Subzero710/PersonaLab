"""Bound and validate ZIP input before reading individual JPEG members."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
import stat
from struct import unpack_from
from typing import BinaryIO
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo
import zlib

from .errors import DatasetError, DFLError, ErrorCode
from .jpeg import MAX_JPEG_BYTES


MAX_ZIP_BYTES = 2**32 - 1
MAX_ZIP_INDEX_BYTES = 8 * 1024**2
MAX_ZIP_ENTRIES = 20_000
MAX_ZIP_EXPANDED_BYTES = 8 * 1024**3
MAX_ZIP_RATIO = 200
_ZIP_EOCD = b"PK\x05\x06"


def _corrupt_zip(message: str) -> DatasetError:
    return DatasetError(ErrorCode.CORRUPT_ZIP, message)


def _preflight(file: BinaryIO) -> None:
    size = os.fstat(file.fileno()).st_size
    if size > MAX_ZIP_BYTES:
        raise DatasetError(ErrorCode.RESOURCE_LIMIT, "ZIP archive exceeds byte limit")

    # The classic ZIP end record follows the central directory, with at most
    # 65,535 comment bytes. Verify the index size before ZipFile allocates it.
    tail_length = min(size, 22 + 65_535)
    file.seek(size - tail_length)
    tail = file.read(tail_length)
    end = tail.rfind(_ZIP_EOCD)
    if end < 0 or len(tail) - end < 22:
        raise _corrupt_zip("Missing ZIP end record")
    disk, index_disk, disk_count, count, index_size, index_offset, comment = unpack_from(
        "<HHHHIIH", tail, end + 4,
    )
    if end + 22 + comment != len(tail):
        raise _corrupt_zip("Truncated ZIP comment or trailing data")
    if disk or index_disk or disk_count != count:
        raise DatasetError(ErrorCode.UNSUPPORTED_INPUT, "Multi-volume ZIP is unsupported")
    if count == 0xFFFF or index_size == 0xFFFFFFFF or index_offset == 0xFFFFFFFF:
        raise DatasetError(ErrorCode.UNSUPPORTED_INPUT, "ZIP64 central directories are unsupported")
    if count > MAX_ZIP_ENTRIES or index_size > MAX_ZIP_INDEX_BYTES:
        raise DatasetError(ErrorCode.RESOURCE_LIMIT, "ZIP central directory exceeds limit")
    index_end = index_offset + index_size
    if index_end != size - tail_length + end:
        raise _corrupt_zip("ZIP central directory offset or size disagrees with end record")

    file.seek(index_offset)
    for _ in range(count):
        header = file.read(46)
        if len(header) != 46 or not header.startswith(b"PK\x01\x02"):
            raise _corrupt_zip("Invalid ZIP central directory header")
        next_record = file.tell() + sum(unpack_from("<HHH", header, 28))
        if next_record > index_end:
            raise _corrupt_zip("ZIP central directory entry crosses index boundary")
        file.seek(next_record)
    if file.tell() != index_end:
        raise _corrupt_zip("ZIP central directory count disagrees with end record")


@contextmanager
def open_zip(file: BinaryIO) -> Iterator[ZipFile]:
    try:
        _preflight(file)
        archive = ZipFile(file, "r")
    except DatasetError:
        raise
    except (BadZipFile, OSError, ValueError) as error:
        raise _corrupt_zip("ZIP central directory cannot be read") from error
    with archive:
        yield archive


def safe_zip_name(name: str) -> bool:
    parts = name.split("/")
    return (
        len(parts) <= 64 and len(name.encode("utf-8")) <= 1_024
        and all(part not in ("", ".", "..") for part in parts)
        and "\\" not in name and ":" not in name
        and all(ord(char) >= 32 and ord(char) != 127 for char in name)
    )


def check_zip_member(info: ZipInfo) -> None:
    if info.flag_bits & (1 | 64):
        raise DFLError(ErrorCode.ENCRYPTED_ZIP_ENTRY, "Encrypted ZIP entries are unsupported")
    kind = stat.S_IFMT(info.external_attr >> 16) if info.create_system == 3 else 0
    if kind not in (0, stat.S_IFREG):
        raise DFLError(ErrorCode.UNSAFE_PATH, "ZIP JPEG entry is not a regular file")
    if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED) or info.flag_bits & (32 | 8192):
        raise DFLError(ErrorCode.UNSUPPORTED_ZIP_ENTRY, "Unsupported ZIP compression or flags")
    if info.volume != 0:
        raise DFLError(ErrorCode.UNSUPPORTED_ZIP_ENTRY, "Multi-volume ZIP member is unsupported")
    if (
        info.file_size > MAX_JPEG_BYTES or info.compress_size > MAX_JPEG_BYTES
        or info.file_size > MAX_ZIP_RATIO * max(1, info.compress_size)
    ):
        raise DFLError(ErrorCode.RESOURCE_LIMIT, "ZIP entry size or expansion ratio exceeds limit")
    if (
        (info.file_size and not info.compress_size)
        or (info.compress_type == ZIP_STORED and info.file_size != info.compress_size)
    ):
        raise DFLError(ErrorCode.CORRUPT_ZIP_ENTRY, "Inconsistent ZIP entry sizes")


def read_zip_member(file: BinaryIO, archive: ZipFile, info: ZipInfo) -> bytes:
    try:
        # ZipFile verifies the member name and that its data does not overlap
        # another record. Check the full compressed stream and declared size.
        with archive.open(info):
            pass
        file.seek(info.header_offset)
        header = file.read(30)
        if len(header) != 30 or not header.startswith(b"PK\x03\x04"):
            raise BadZipFile("Invalid ZIP local header")
        flags, method, _, _, crc, compressed, uncompressed, name_length, extra_length = unpack_from(
            "<HHHHIIIHH", header, 6,
        )
        if flags & (1 | 64):
            raise DFLError(ErrorCode.ENCRYPTED_ZIP_ENTRY, "Encrypted ZIP local header")
        if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF:
            raise DFLError(ErrorCode.UNSUPPORTED_ZIP_ENTRY, "ZIP64 local headers are unsupported")
        if flags != info.flag_bits or method != info.compress_type:
            raise BadZipFile("ZIP local header disagrees with central directory")
        if not flags & 8 and (crc, compressed, uncompressed) != (
            info.CRC, info.compress_size, info.file_size,
        ):
            raise BadZipFile("ZIP local sizes or CRC disagree with central directory")
        file.seek(name_length + extra_length, os.SEEK_CUR)
        compressed_data = file.read(info.compress_size)
        if len(compressed_data) != info.compress_size:
            raise BadZipFile("Truncated ZIP member")
        if method == ZIP_DEFLATED:
            decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            data = decoder.decompress(compressed_data, info.file_size + 1)
            if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
                raise BadZipFile("ZIP DEFLATE data does not end within member bounds")
        else:
            data = compressed_data
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise BadZipFile("ZIP entry length or CRC differs from its header")
        return data
    except (BadZipFile, OSError, EOFError, RuntimeError, UnicodeError, zlib.error) as error:
        raise DFLError(ErrorCode.CORRUPT_ZIP_ENTRY, "ZIP entry failed header, CRC or decompression checks") from error
