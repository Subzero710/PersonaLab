"""Exercise rejection of unsafe data without external actions or payloads."""

import pickle
import struct

import numpy as np
import pytest

from personalab.dfl import ErrorCode, SampleError, read_jpeg


_CALLS = []


def _record_call():
    _CALLS.append("unexpected call")
    return None


class _FixtureWithReducer:
    def __reduce__(self):
        return _record_call, ()


class _ArrayWithInvalidState:
    def __init__(self, shape):
        self.shape = shape

    def __reduce__(self):
        reducer, args, (version, _, dtype, fortran, data) = np.zeros(4, np.uint8).__reduce__()
        return reducer, args, (version, self.shape, dtype, fortran, data)


def test_rejects_foreign_callable_before_any_execution(dfl_file, dfl_points, monkeypatch):
    _CALLS.clear()
    path = dfl_file({"landmarks": dfl_points, "fixture": _FixtureWithReducer()})

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unrestricted pickle loader was invoked")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    with pytest.raises(SampleError) as rejected:
        read_jpeg(path)
    assert rejected.value.code == ErrorCode.UNSAFE_SERIALIZATION
    assert _CALLS == []


@pytest.mark.parametrize(
    "payload",
    [
        b"\x80\x02cfixture_only\nunknown_value\n.",
        b"\x80\x02cnumpy\nndarray\n)\x81.",  # NEWOBJ
        b"\x80\x02cnumpy\nndarray\n)R.",     # unavailable REDUCE target
        b"\x80\x02}Nb.",                    # BUILD on a non-token
        b"\x80\x02Q.",                      # BINPERSID
        b"\x80\x02Pfixture\n.",             # PERSID
        b"\x80\x02\x82\x01.",               # EXT1
        b"\x80\x02\x83\x01\x00.",           # EXT2
        b"\x80\x02\x84\x01\x00\x00\x00.",   # EXT4
        b"\x80\x05\x97.",                   # NEXT_BUFFER
        b"\x80\x05\x98.",                   # READONLY_BUFFER
    ],
)
def test_forbidden_pickle_operations(dfl_file, payload):
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file(None, app15=(payload,)))
    assert rejected.value.code == ErrorCode.UNSAFE_SERIALIZATION


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (np.array([0], dtype=object), ErrorCode.UNSUPPORTED_METADATA_TYPE),
        (np.array([(1, 2)], dtype=[("a", "i1"), ("b", "u1")]), ErrorCode.UNSUPPORTED_METADATA_TYPE),
        (_ArrayWithInvalidState((-1,)), ErrorCode.MALFORMED_APP15),
        (_ArrayWithInvalidState((10,)), ErrorCode.MALFORMED_APP15),
        (_ArrayWithInvalidState((1, 1, 1, 1, 4)), ErrorCode.RESOURCE_LIMIT),
        (_ArrayWithInvalidState((10_000, 10_000)), ErrorCode.RESOURCE_LIMIT),
    ],
)
def test_invalid_numpy_layouts_fail_before_reconstruction(dfl_file, dfl_points, value, code):
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, "fixture": value}))
    assert rejected.value.code == code


def test_framed_opcodes_cannot_cross_boundaries(dfl_file, dfl_points):
    payload = bytearray(pickle.dumps({"landmarks": dfl_points}, protocol=4))
    assert payload[2] == 0x95
    struct.pack_into("<Q", payload, 3, 4)  # End inside the first string opcode.
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file(None, app15=(payload,)))
    assert rejected.value.code == ErrorCode.MALFORMED_APP15


def test_pickle_memo_index_is_bounded(dfl_file):
    payload = b"\x80\x02}r" + struct.pack("<I", 2**31) + b"."
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file(None, app15=(payload,)))
    assert rejected.value.code == ErrorCode.RESOURCE_LIMIT


@pytest.mark.parametrize("duplicate", [False, True])
def test_nesting_and_alias_expansion_are_bounded(dfl_file, dfl_points, duplicate):
    value = 0
    for _ in range(20 if duplicate else 40):
        value = [value, value] if duplicate else [value]
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, "fixture": value}))
    assert rejected.value.code == ErrorCode.RESOURCE_LIMIT


def test_cyclic_metadata_is_rejected(dfl_file, dfl_points):
    cycle = []
    cycle.append(cycle)
    with pytest.raises(SampleError) as rejected:
        read_jpeg(dfl_file({"landmarks": dfl_points, "fixture": cycle}))
    assert rejected.value.code == ErrorCode.UNSAFE_SERIALIZATION
