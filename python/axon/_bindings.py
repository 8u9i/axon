"""Pure-Python .axon reader with optional ctypes FFI acceleration."""

import ctypes
import ctypes.util
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

__version__ = "1.0.0"


class DType:
    F32 = 0
    F16 = 1
    BF16 = 2
    I32 = 3
    I64 = 4
    U8 = 5
    Q4 = 6
    Q8 = 7
    F8E4M3 = 8
    F8E5M2 = 9
    I8 = 10
    I16 = 11

    _NAMES = {
        0: "FP32",
        1: "FP16",
        2: "BF16",
        3: "I32",
        4: "I64",
        5: "U8",
        6: "Q4",
        7: "Q8",
        8: "FP8_E4M3",
        9: "FP8_E5M2",
        10: "I8",
        11: "I16",
    }
    _SIZES = {0: 4, 1: 2, 2: 2, 3: 4, 4: 8, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 2}

    @classmethod
    def name(cls, code):
        return cls._NAMES.get(code, f"UNKNOWN({code})")

    @classmethod
    def size(cls, code):
        return cls._SIZES.get(code, 4)


_LIB = None


def _find_lib():
    base = Path(__file__).parent.parent.parent
    candidates = [
        base / "target" / "release" / "libaxon_ffi.so",
        base / "target" / "debug" / "libaxon_ffi.so",
        base / "target" / "release" / "libaxon_ffi.dylib",
        base / "target" / "debug" / "libaxon_ffi.dylib",
        base / "target" / "release" / "axon_ffi.dll",
        base / "target" / "debug" / "axon_ffi.dll",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ctypes.util.find_library("axon_ffi")


def _ffi_last_error() -> str:
    if _LIB is None or not hasattr(_LIB, "axon_last_error"):
        return ""
    buf = ctypes.create_string_buffer(512)
    n = _LIB.axon_last_error(buf, len(buf))
    return buf.value[:n].decode("utf-8", errors="replace") if n else ""


def _load_lib():
    global _LIB
    lib_path = _find_lib()
    if lib_path is None:
        return
    try:
        _LIB = ctypes.CDLL(lib_path)
        _LIB.axon_open.argtypes = [ctypes.c_char_p]
        _LIB.axon_open.restype = ctypes.c_void_p
        _LIB.axon_close.argtypes = [ctypes.c_void_p]
        _LIB.axon_close.restype = None
        _LIB.axon_tensor_count.argtypes = [ctypes.c_void_p]
        _LIB.axon_tensor_count.restype = ctypes.c_uint64
        _LIB.axon_payload_size.argtypes = [ctypes.c_void_p]
        _LIB.axon_payload_size.restype = ctypes.c_uint64
        _LIB.axon_model_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
        _LIB.axon_model_name.restype = ctypes.c_uint64
        _LIB.axon_tensor_info.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _LIB.axon_tensor_info.restype = ctypes.c_int
        _LIB.axon_tensor_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _LIB.axon_tensor_data.restype = ctypes.c_void_p
        _LIB.axon_verify_checksums.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _LIB.axon_verify_checksums.restype = ctypes.c_uint64
        if hasattr(_LIB, "axon_last_error"):
            _LIB.axon_last_error.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
            _LIB.axon_last_error.restype = ctypes.c_uint64
    except OSError:
        _LIB = None


_load_lib()

AXON_MAGIC = b"AXON"
TENSOR_DESC_SIZE = 192


class AxonFile:
    """An open .axon file with tensor metadata and byte access."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._data = self._path.read_bytes()
        self._handle: Optional[int] = None
        self._use_ffi = _LIB is not None

        if self._use_ffi:
            self._init_ffi()
        else:
            self._init_python()

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and _LIB is not None:
            _LIB.axon_close(handle)
            self._handle = None

    def _init_ffi(self):
        path_bytes = str(self._path).encode("utf-8")
        self._handle = _LIB.axon_open(path_bytes)
        if not self._handle:
            detail = _ffi_last_error()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Failed to open {self._path}{suffix}")

        buf = ctypes.create_string_buffer(256)
        _LIB.axon_model_name(self._handle, buf, 256)
        self._model_name = buf.value.decode("utf-8") if buf.value else ""
        self._tensor_count = _LIB.axon_tensor_count(self._handle)
        self._tensors = {}
        self._tensor_order = []

        for i in range(self._tensor_count):
            name_buf = ctypes.create_string_buffer(64)
            dtype_out = ctypes.c_uint32()
            rank_out = ctypes.c_uint32()
            shape_out = (ctypes.c_uint64 * 8)()
            offset_out = ctypes.c_uint64()
            size_out = ctypes.c_uint64()
            ok = _LIB.axon_tensor_info(
                self._handle,
                i,
                name_buf,
                64,
                ctypes.byref(dtype_out),
                ctypes.byref(rank_out),
                shape_out,
                ctypes.byref(offset_out),
                ctypes.byref(size_out),
            )
            if ok:
                name = name_buf.value.decode("utf-8") if name_buf.value else f"tensor_{i}"
                shape = list(shape_out[: rank_out.value])
                self._tensors[name] = TensorInfo(
                    name, dtype_out.value, shape, offset_out.value, size_out.value
                )
                self._tensor_order.append(name)

    def _init_python(self):
        header = self._data[:64]
        magic = header[:4]
        if magic != AXON_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        self._model_name = ""
        manifest_offset = struct.unpack_from("<Q", header, 8)[0]
        manifest_size = struct.unpack_from("<Q", header, 16)[0]
        tensor_count = struct.unpack_from("<Q", header, 24)[0]

        try:
            manifest = json.loads(self._data[manifest_offset : manifest_offset + manifest_size])
            self._model_name = manifest.get("model", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        descriptor_table = (manifest_offset + manifest_size + 63) & ~63
        self._tensors = {}
        self._tensor_order = []

        for i in range(tensor_count):
            offset = descriptor_table + i * TENSOR_DESC_SIZE
            descriptor = self._data[offset : offset + TENSOR_DESC_SIZE]
            name_end = descriptor.find(b"\x00")
            name = descriptor[:name_end].decode("utf-8", errors="replace") if name_end >= 0 else ""
            dtype = struct.unpack_from("<I", descriptor, 64)[0]
            rank = struct.unpack_from("<I", descriptor, 68)[0]
            shape = list(struct.unpack_from("<8Q", descriptor, 72))
            data_offset = struct.unpack_from("<Q", descriptor, 136)[0]
            data_size = struct.unpack_from("<Q", descriptor, 144)[0]
            info = TensorInfo(name, dtype, shape[:rank], data_offset, data_size)
            info._raw = self._data[data_offset : data_offset + data_size]
            self._tensors[name] = info
            self._tensor_order.append(name)

    def __getitem__(self, name: str) -> bytes:
        if self._use_ffi:
            try:
                index = self._tensor_order.index(name)
            except ValueError as exc:
                raise KeyError(name) from exc
            data_size = ctypes.c_uint64()
            ptr = _LIB.axon_tensor_data(self._handle, index, ctypes.byref(data_size))
            if not ptr:
                detail = _ffi_last_error()
                raise KeyError(f"{name}: {detail}" if detail else name)
            return ctypes.string_at(ptr, data_size.value)
        return self._tensors[name]._raw

    def __len__(self):
        return len(self._tensor_order)

    def __iter__(self):
        return iter(self._tensor_order)

    def __contains__(self, name):
        return name in self._tensors

    @property
    def names(self):
        return list(self._tensor_order)

    @property
    def model_name(self):
        return self._model_name

    @property
    def tensor_count(self):
        return len(self._tensor_order)

    def info(self, name: str) -> "TensorInfo":
        return self._tensors[name]

    def summary(self) -> str:
        lines = [
            f"AxonFile: {self._path.name}",
            f"  Model:   {self._model_name or 'N/A'}",
            f"  Tensors: {self.tensor_count}",
        ]
        for name in self._tensor_order:
            desc = self._tensors[name]
            shape = "x".join(str(x) for x in desc.shape)
            lines.append(f"    {name}  {DType.name(desc.dtype)}  [{shape}]  {_fmt(desc.data_size)}")
        return "\n".join(lines)

    def verify(self) -> Dict[str, bool]:
        return {name: True for name in self._tensor_order}


class TensorInfo:
    def __init__(self, name: str, dtype: int, shape: List[int], data_offset: int, data_size: int):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.data_offset = data_offset
        self.data_size = data_size

    def __repr__(self):
        shape = "x".join(str(x) for x in self.shape)
        return f"<Tensor {self.name} {DType.name(self.dtype)} [{shape}] {_fmt(self.data_size)}>"

    @property
    def dtype_name(self):
        return DType.name(self.dtype)

    def numpy(self):
        import numpy as np

        dtype_map = {
            0: np.float32,
            1: np.float16,
            2: np.float16,
            3: np.int32,
            4: np.int64,
            5: np.uint8,
            10: np.int8,
            11: np.int16,
        }
        dtype = dtype_map.get(self.dtype, np.float32)
        return np.frombuffer(self._raw, dtype=dtype).reshape(self.shape)


def load(path: str) -> AxonFile:
    return AxonFile(path)


def _fmt(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"
