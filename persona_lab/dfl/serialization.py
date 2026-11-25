"""Read the data-only subset of binary pickle used in DFL APP15 segments."""

from dataclasses import dataclass
from enum import Enum, auto
from math import prod
import pickletools
from struct import error as StructError

import numpy as np

from .errors import DFLError, ErrorCode


MAX_PAYLOAD_BYTES = 65_533
MAX_OPERATIONS = 16_384
MAX_MEMO_ENTRIES = 4_096
MAX_DEPTH = 32
MAX_ARRAY_AXES = 4
MAX_ARRAY_BYTES = 4 * MAX_PAYLOAD_BYTES
_DTYPE_CODES = {
    "b1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8",
}


class _Symbol(Enum):
    NDARRAY = auto()
    DTYPE = auto()
    RECONSTRUCT = auto()
    FROMBUFFER = auto()
    SCALAR = auto()
    LATIN1 = auto()
    EMPTY_BYTES = auto()


# Resolve names to inert symbols. No payload name is imported or called.
_SYMBOLS = {
    ("numpy", "ndarray"): _Symbol.NDARRAY,
    ("numpy", "dtype"): _Symbol.DTYPE,
    ("numpy.core._multiarray_umath", "dtype"): _Symbol.DTYPE,
    ("numpy._core._multiarray_umath", "dtype"): _Symbol.DTYPE,
    ("numpy.core.multiarray", "_reconstruct"): _Symbol.RECONSTRUCT,
    ("numpy._core.multiarray", "_reconstruct"): _Symbol.RECONSTRUCT,
    ("numpy._core._multiarray_umath", "_reconstruct"): _Symbol.RECONSTRUCT,
    ("numpy.core.multiarray", "scalar"): _Symbol.SCALAR,
    ("numpy._core.multiarray", "scalar"): _Symbol.SCALAR,
    ("numpy._core._multiarray_umath", "scalar"): _Symbol.SCALAR,
    ("numpy.core.numeric", "_frombuffer"): _Symbol.FROMBUFFER,
    ("numpy._core.numeric", "_frombuffer"): _Symbol.FROMBUFFER,
    ("_codecs", "encode"): _Symbol.LATIN1,
    ("__builtin__", "bytes"): _Symbol.EMPTY_BYTES,
    ("builtins", "bytes"): _Symbol.EMPTY_BYTES,
}


@dataclass
class _DType:
    code: str
    value: np.dtype | None = None


@dataclass
class _Array:
    value: np.ndarray | None = None


def _symbol(module: object, name: object) -> _Symbol:
    if type(module) is str and type(name) is str:
        result = _SYMBOLS.get((module, name))
        if result is not None:
            return result
    raise DFLError(ErrorCode.UNSAFE_SERIALIZATION, "Pickle global is forbidden")


def _build_dtype(target: _DType, state: tuple) -> None:
    if (
        len(state) != 8 or type(state[0]) is not int or state[0] != 3
        or type(state[1]) is not str or any(item is not None for item in state[2:5])
        or any(type(item) is not int for item in state[5:])
        or state[5:] != (-1, -1, 0)
    ):
        raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Unsupported NumPy dtype state")
    dtype = np.dtype(target.code)
    order = state[1]
    if len(order) != 1 or order not in ("|" if dtype.itemsize == 1 else "<>"):
        raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Invalid NumPy byte order")
    target.value = dtype.newbyteorder(order)


class _Decoder:
    def __init__(self):
        self.stack: list = []
        self.marks: list[int] = []
        self.memo: dict[int, object] = {}
        self.array_bytes = 0
        self.buffer_bytes = 0
        self.values = 0

    def _take(self, count: int) -> list:
        start = len(self.stack) - count
        if start < (self.marks[-1] if self.marks else 0):
            raise DFLError(ErrorCode.MALFORMED_APP15, "Incomplete pickle operands")
        result = self.stack[start:]
        del self.stack[start:]
        return result

    def _top(self):
        if len(self.stack) <= (self.marks[-1] if self.marks else 0):
            raise DFLError(ErrorCode.MALFORMED_APP15, "Missing pickle operand")
        return self.stack[-1]

    def _take_mark(self) -> list:
        if not self.marks:
            raise DFLError(ErrorCode.MALFORMED_APP15, "Missing pickle mark")
        return self._take(len(self.stack) - self.marks.pop())

    def _array(self, data, dtype, shape, order) -> np.ndarray:
        if type(dtype) is not _DType or dtype.value is None or type(shape) is not tuple:
            raise DFLError(ErrorCode.MALFORMED_APP15, "Missing NumPy dtype or shape")
        if len(shape) > MAX_ARRAY_AXES:
            raise DFLError(ErrorCode.RESOURCE_LIMIT, "NumPy array has too many axes")
        for size in shape:
            if type(size) is not int or size < 0:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Invalid NumPy array axis")
            if size > MAX_PAYLOAD_BYTES:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "NumPy array axis exceeds limit")
        size = prod(shape) * dtype.value.itemsize
        self.array_bytes += size
        if size > MAX_PAYLOAD_BYTES or self.array_bytes > MAX_ARRAY_BYTES:
            raise DFLError(ErrorCode.RESOURCE_LIMIT, "NumPy array data exceeds limit")
        if type(data) is not bytes or len(data) != size:
            raise DFLError(ErrorCode.MALFORMED_APP15, "NumPy array buffer size disagrees with shape")
        return np.frombuffer(data, dtype=dtype.value).reshape(shape, order=order)

    def _reduce(self, symbol, args):
        if type(args) is not tuple:
            raise DFLError(ErrorCode.MALFORMED_APP15, "Pickle reduction requires a tuple")
        if symbol is _Symbol.DTYPE and len(args) == 3:
            code, align, copy = args
            # Legacy NumPy pickles encode these two flags as integers 0 and 1.
            modern_flags = align is False and copy is True
            legacy_flags = (
                type(align) is int and align == 0
                and type(copy) is int and copy == 1
            )
            if type(code) is str and code in _DTYPE_CODES and (modern_flags or legacy_flags):
                return _DType(code)
            raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Unsupported NumPy dtype")
        if symbol is _Symbol.RECONSTRUCT and len(args) == 3:
            cls, shape, typecode = args
            if (
                cls is _Symbol.NDARRAY and type(shape) is tuple and len(shape) == 1
                and type(shape[0]) is int and shape[0] == 0
                and type(typecode) is bytes and typecode == b"b"
            ):
                return _Array()
        if symbol is _Symbol.FROMBUFFER and len(args) == 4:
            data, dtype, shape, order = args
            if type(order) is str and order in ("C", "F"):
                return self._array(data, dtype, shape, order)
        if symbol is _Symbol.SCALAR and len(args) == 2:
            dtype, data = args
            return self._array(data, dtype, (), "C")[()]
        if symbol is _Symbol.LATIN1 and len(args) == 2:
            value, encoding = args
            if type(value) is str and type(encoding) is str and encoding == "latin1":
                self.buffer_bytes += len(value)
                if self.buffer_bytes > MAX_ARRAY_BYTES:
                    raise DFLError(ErrorCode.RESOURCE_LIMIT, "Legacy byte buffers exceed limit")
                return value.encode("latin1")
        if symbol is _Symbol.EMPTY_BYTES and not args:
            return b""
        raise DFLError(ErrorCode.UNSAFE_SERIALIZATION, "Pickle reduction is forbidden")

    def _build(self, target, state) -> None:
        if type(state) is tuple and type(target) is _DType and target.value is None:
            _build_dtype(target, state)
            return
        if type(state) is tuple and type(target) is _Array and target.value is None:
            if len(state) != 5 or type(state[0]) is not int or state[0] != 1:
                raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Unsupported NumPy array state")
            _, shape, dtype, fortran, data = state
            if type(fortran) is not bool:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Invalid NumPy array order")
            target.value = self._array(data, dtype, shape, "F" if fortran else "C")
            return
        raise DFLError(ErrorCode.UNSAFE_SERIALIZATION, "Pickle BUILD target is forbidden")

    @staticmethod
    def _set_items(target: dict, items: list) -> None:
        if type(target) is not dict or len(items) % 2:
            raise DFLError(ErrorCode.MALFORMED_APP15, "Invalid pickle dictionary operands")
        for key, value in zip(items[::2], items[1::2]):
            if type(key) is not str:
                raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Metadata keys must be strings")
            if key in target:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Duplicate metadata dictionary key")
            target[key] = value

    def _store(self, index: int) -> None:
        if not 0 <= index < MAX_MEMO_ENTRIES or len(self.memo) >= MAX_MEMO_ENTRIES:
            raise DFLError(ErrorCode.RESOURCE_LIMIT, "Pickle memo exceeds limit")
        if index in self.memo:
            raise DFLError(ErrorCode.MALFORMED_APP15, "Pickle memo entry was redefined")
        self.memo[index] = self._top()

    def _apply(self, name: str, arg) -> None:
        if name in (
            "BININT", "BININT1", "BININT2", "LONG1", "LONG4", "BINFLOAT",
            "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
            "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
        ):
            if type(arg) is int and arg.bit_length() > 64:
                raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Metadata integer exceeds 64 bits")
            self.stack.append(arg)
        elif name == "BYTEARRAY8":
            self.stack.append(bytes(arg))
        elif name in ("NONE", "NEWTRUE", "NEWFALSE"):
            self.stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
        elif name == "MARK":
            if len(self.marks) >= MAX_DEPTH:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "Pickle nesting exceeds limit")
            self.marks.append(len(self.stack))
        elif name in ("EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE"):
            self.stack.append({"EMPTY_DICT": {}, "EMPTY_LIST": [], "EMPTY_TUPLE": ()}[name])
        elif name in ("LIST", "TUPLE", "DICT"):
            items = self._take_mark()
            if name == "DICT":
                mapping: dict = {}
                self._set_items(mapping, items)
                self.stack.append(mapping)
            else:
                self.stack.append(items if name == "LIST" else tuple(items))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            self.stack.append(tuple(self._take(int(name[-1]))))
        elif name in ("APPEND", "APPENDS"):
            items = self._take(1) if name == "APPEND" else self._take_mark()
            target = self._top()
            if type(target) is not list:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Pickle append target must be a list")
            target.extend(items)
        elif name in ("SETITEM", "SETITEMS"):
            items = self._take(2) if name == "SETITEM" else self._take_mark()
            self._set_items(self._top(), items)
        elif name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
            self._store(len(self.memo) if name == "MEMOIZE" else arg)
        elif name in ("BINGET", "LONG_BINGET"):
            if arg not in self.memo:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Missing pickle memo entry")
            self.stack.append(self.memo[arg])
        elif name == "GLOBAL":
            self.stack.append(_symbol(*arg.split(" ", 1)))
        elif name == "STACK_GLOBAL":
            self.stack.append(_symbol(*self._take(2)))
        elif name == "REDUCE":
            self.stack.append(self._reduce(*self._take(2)))
        elif name == "BUILD":
            state, = self._take(1)
            self._build(self._top(), state)
        else:
            raise DFLError(ErrorCode.UNSAFE_SERIALIZATION, "Pickle opcode is forbidden")

    def _materialize(self, value, active: set[int]):
        self.values += 1
        if self.values > MAX_OPERATIONS or len(active) > MAX_DEPTH:
            raise DFLError(ErrorCode.RESOURCE_LIMIT, "Decoded metadata exceeds nesting or value limit")
        if type(value) is _Array:
            if value.value is None:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Incomplete NumPy array")
            return value.value
        if type(value) is _Symbol:
            raise DFLError(ErrorCode.UNSAFE_SERIALIZATION, "Metadata contains a pickle global")
        if type(value) is _DType:
            raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Metadata contains a standalone dtype")
        if type(value) not in (dict, list, tuple):
            return value
        if id(value) in active:
            raise DFLError(ErrorCode.UNSAFE_SERIALIZATION, "Cyclic metadata is forbidden")
        active.add(id(value))
        try:
            if type(value) is dict:
                return {key: self._materialize(item, active) for key, item in value.items()}
            items = [self._materialize(item, active) for item in value]
            return items if type(value) is list else tuple(items)
        finally:
            active.remove(id(value))

    def decode(self, data: bytes) -> dict:
        frame_end = None
        for count, (opcode, arg, position) in enumerate(pickletools.genops(data), 1):
            if count > MAX_OPERATIONS:
                raise DFLError(ErrorCode.RESOURCE_LIMIT, "Pickle opcode count exceeds limit")
            name = opcode.name
            if count == 1:
                if name != "PROTO" or not 2 <= arg <= 5:
                    raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Expected binary pickle protocol 2–5")
                protocol = arg
                continue
            if name == "PROTO" or opcode.proto > protocol:
                raise DFLError(ErrorCode.MALFORMED_APP15, "Invalid pickle protocol or opcode")
            if frame_end is not None and position >= frame_end:
                if position != frame_end:
                    raise DFLError(ErrorCode.MALFORMED_APP15, "Pickle opcode crosses a frame boundary")
                frame_end = None
            if name == "FRAME":
                if frame_end is not None or position + 9 + arg > len(data):
                    raise DFLError(ErrorCode.MALFORMED_APP15, "Invalid pickle frame length")
                frame_end = position + 9 + arg
            elif name == "STOP":
                if position + 1 != len(data) or self.marks or len(self.stack) != 1:
                    raise DFLError(ErrorCode.MALFORMED_APP15, "Invalid pickle termination")
                if type(self.stack[0]) is not dict:
                    raise DFLError(ErrorCode.UNSUPPORTED_METADATA_TYPE, "Metadata root must be a dictionary")
                return self._materialize(self.stack[0], set())
            else:
                self._apply(name, arg)
        raise DFLError(ErrorCode.MALFORMED_APP15, "Pickle is missing STOP")


def decode_metadata(data: bytes) -> dict:
    """Decode bounded pickle data without invoking the pickle virtual machine."""
    if len(data) > MAX_PAYLOAD_BYTES:
        raise DFLError(ErrorCode.RESOURCE_LIMIT, "APP15 payload exceeds limit")
    try:
        return _Decoder().decode(data)
    except DFLError:
        raise
    except (ValueError, OverflowError, StructError) as error:
        raise DFLError(ErrorCode.MALFORMED_APP15, "Malformed pickle data") from error
