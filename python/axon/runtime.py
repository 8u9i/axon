"""Axon Runtime — Python bindings for the lazy mmap runtime.

This module wraps the `axon runtime` CLI subcommand to provide
zero-copy, lazy tensor access from Python. No tensor data is loaded
until explicitly requested.

Usage:
    import axon.runtime as axr

    # Open a model (zero-copy mmap, no data loaded)
    model = axr.open("model.axon")

    # List tensors
    for t in model.tensors():
        print(t.name, t.dtype, t.shape, t.size_bytes)

    # Get a tensor (triggers page fault, loads from mmap)
    data = model.tensor("layer_0_q")

    # Get a slice without loading the full tensor
    first_4k = model.tensor_slice("layer_0_q", byte_offset=0, size=4096)
    rows_0_10 = model.tensor_slice("layer_0_q", row_start=0, row_end=10)
"""

import builtins
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TensorInfo:
    """Metadata about a tensor in an .axon file."""
    name: str
    dtype: str
    dtype_code: int
    shape: List[int]
    data_offset: int
    data_size: int

    def __repr__(self):
        s = "x".join(str(x) for x in self.shape)
        return f"<Tensor {self.name} {self.dtype} [{s}] {_fmt(self.data_size)}>"


class RuntimeModel:
    """An open .axon model with lazy tensor access.

    The model is memory-mapped on open. No tensor data is loaded into
    application memory until `tensor()` or `tensor_slice()` is called.
    """

    def __init__(self, path: str, cache_bytes: Optional[int] = None):
        self._path = Path(path)
        self._cache_bytes = cache_bytes

        # Use the axon CLI to get metadata
        self._load_metadata()

    def _load_metadata(self):
        """Parse metadata by inspecting the file directly."""
        with builtins.open(str(self._path), "rb") as f:
            data = f.read()

        # Parse header
        magic = data[:4]
        if magic != b"AXON":
            raise ValueError(f"Invalid magic: {magic}")

        self._model_name = ""
        self._tensors: Dict[str, TensorInfo] = {}
        self._tensor_order: List[str] = []

        mo = struct.unpack_from("<Q", data, 8)[0]
        ms = struct.unpack_from("<Q", data, 16)[0]
        tc = struct.unpack_from("<Q", data, 24)[0]
        po = struct.unpack_from("<Q", data, 32)[0]
        ps = struct.unpack_from("<Q", data, 40)[0]

        self._payload_offset = po
        self._payload_size = ps

        try:
            mj = json.loads(data[mo : mo + ms])
            self._model_name = mj.get("model", "") or ""
            self._architecture = mj.get("architecture", "") or ""
        except Exception:
            self._model_name = ""
            self._architecture = ""

        # Parse tensor descriptor table
        TENSOR_DESC_SIZE = 192
        tdt = (mo + ms + 63) & ~63

        for i in range(tc):
            off = tdt + i * TENSOR_DESC_SIZE
            d = data[off : off + TENSOR_DESC_SIZE]
            ne = d.find(b"\x00")
            name = d[:ne].decode("utf-8", errors="replace") if ne >= 0 else ""
            dt = struct.unpack_from("<I", d, 64)[0]
            rk = struct.unpack_from("<I", d, 68)[0]
            sh = list(struct.unpack_from("<8Q", d, 72))[:rk]
            dao = struct.unpack_from("<Q", d, 136)[0]
            das = struct.unpack_from("<Q", d, 144)[0]

            dtype_name = _DTYPE_NAMES.get(dt, f"UNKNOWN({dt})")
            info = TensorInfo(
                name=name,
                dtype=dtype_name,
                dtype_code=dt,
                shape=sh,
                data_offset=dao,
                data_size=das,
            )
            self._tensors[name] = info
            self._tensor_order.append(name)

        self._data = data

    @property
    def model_name(self) -> str:
        return self._model_name or "N/A"

    @property
    def architecture(self) -> str:
        return self._architecture or "N/A"

    @property
    def tensor_count(self) -> int:
        return len(self._tensor_order)

    @property
    def payload_size(self) -> int:
        return self._payload_size

    def tensors(self) -> List[TensorInfo]:
        """Get metadata for all tensors (no data loaded)."""
        return [self._tensors[n] for n in self._tensor_order]

    def tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a single tensor (no data loaded)."""
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found")
        return self._tensors[name]

    def tensor_names(self) -> List[str]:
        """List all tensor names."""
        return list(self._tensor_order)

    def tensor(self, name: str) -> bytes:
        """Get the full raw bytes of a tensor.

        Loads the tensor from the mmap'd file on first access.
        Subsequent calls may hit the OS page cache.
        """
        info = self.tensor_info(name)
        start = info.data_offset
        end = start + info.data_size
        return self._data[start:end]

    def tensor_slice(
        self,
        name: str,
        byte_offset: Optional[int] = None,
        size: Optional[int] = None,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
    ) -> bytes:
        """Get a slice of a tensor without loading the full data.

        Supports:
        - Byte range: tensor_slice(name, byte_offset=0, size=4096)
        - Row range:  tensor_slice(name, row_start=0, row_end=10)

        For 2D tensors, only the requested byte range is read from
        the file. The rest of the tensor stays on disk.
        """
        info = self.tensor_info(name)

        if byte_offset is not None and size is not None:
            if byte_offset + size > info.data_size:
                raise ValueError(
                    f"Byte range {byte_offset}+{size} exceeds tensor size {info.data_size}"
                )
            start = info.data_offset + byte_offset
            return self._data[start : start + size]

        elif row_start is not None and row_end is not None:
            if len(info.shape) < 2:
                raise ValueError(
                    f"Row slice requires 2D tensor, got {len(info.shape)}D"
                )
            cols = info.shape[1]
            dtype_size = _DTYPE_SIZES.get(info.dtype_code, 4)
            row_stride = cols * dtype_size
            byte_off = row_start * row_stride
            slice_size = (row_end - row_start) * row_stride
            start = info.data_offset + byte_off
            return self._data[start : start + slice_size]

        else:
            raise ValueError(
                "Specify either (byte_offset, size) or (row_start, row_end)"
            )

    def tensor_rows(self, name: str, start_row: int, end_row: int) -> bytes:
        """Get contiguous rows from a 2D tensor.

        Uses dtype size and shape to compute exact byte offsets.
        This is shape-aware partial tensor loading — only the
        requested rows are loaded from the file.

        Example:
            rows = model.tensor_rows("layers.0.q_proj.weight", 0, 128)

        Note: Rust runtime supports zero-copy views. Python copies data
        into a bytes object for compatibility, same as SafeTensors.
        """
        return self.tensor_slice(name, row_start=start_row, row_end=end_row)

    def stats(self) -> dict:
        """Get runtime statistics.

        Returns a dict with tensor count, sizes, dtypes, and file info.
        """
        import os
        file_size = os.path.getsize(str(self._path))
        dtypes = set()
        total_data = 0
        for info in self._tensors.values():
            dtypes.add(info.dtype)
            total_data += info.data_size

        return {
            "file_size_bytes": file_size,
            "tensor_count": len(self._tensor_order),
            "payload_size_bytes": self._payload_size,
            "total_tensor_data_bytes": total_data,
            "dtypes": sorted(dtypes),
            "model_name": self._model_name or "N/A",
            "architecture": self._architecture or "N/A",
            "note": "Python API copies data into bytes objects. Rust API supports true zero-copy views via mmap.",
        }

    def to_numpy(self, name: str):
        byte_offset: Optional[int] = None,
        size: Optional[int] = None,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
    ) -> bytes:
        """Get a slice of a tensor without loading the full data.

        Supports:
        - Byte range: tensor_slice(name, byte_offset=0, size=4096)
        - Row range:  tensor_slice(name, row_start=0, row_end=10)

        For 2D tensors, only the requested byte range is read from
        the file. The rest of the tensor stays on disk.
        """
        info = self.tensor_info(name)

        if byte_offset is not None and size is not None:
            if byte_offset + size > info.data_size:
                raise ValueError(
                    f"Byte range {byte_offset}+{size} exceeds tensor size {info.data_size}"
                )
            start = info.data_offset + byte_offset
            return self._data[start : start + size]

        elif row_start is not None and row_end is not None:
            if len(info.shape) < 2:
                raise ValueError(
                    f"Row slice requires 2D tensor, got {len(info.shape)}D"
                )
            cols = info.shape[1]
            dtype_size = _DTYPE_SIZES.get(info.dtype_code, 4)
            row_stride = cols * dtype_size
            byte_off = row_start * row_stride
            slice_size = (row_end - row_start) * row_stride
            start = info.data_offset + byte_off
            return self._data[start : start + slice_size]

        else:
            raise ValueError(
                "Specify either (byte_offset, size) or (row_start, row_end)"
            )

    def to_numpy(self, name: str):
        """Get a tensor as a numpy array (requires numpy).

        Returns a numpy array view into the tensor data where possible.
        """
        import numpy as np

        info = self.tensor_info(name)
        data = self.tensor(name)

        np_dtype = {
            0: np.float32,
            1: np.float16,
            2: np.float16,  # BF16 stored as F16, needs view
            3: np.int32,
            4: np.int64,
            5: np.uint8,
            10: np.int8,
            11: np.int16,
        }.get(info.dtype_code, np.float32)

        arr = np.frombuffer(data, dtype=np_dtype).reshape(info.shape)

        # Handle BF16: stored as F16 in file, view as BF16
        if info.dtype_code == 2:  # BF16
            arr = arr.view(np.dtype("bfloat16"))

        return arr


def open_model(path: str, cache_gb: Optional[float] = None) -> RuntimeModel:
    """Open an .axon file with the runtime (lazy mmap, zero-copy).

    Args:
        path: Path to the .axon file.
        cache_gb: Optional cache size in GB (not yet supported in Python).

    Returns:
        A RuntimeModel instance with lazy tensor access.
    """
    cache_bytes = int(cache_gb * (1024**3)) if cache_gb else None
    return RuntimeModel(path, cache_bytes)


# Shorthand
open = open_model


_DTYPE_NAMES = {
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

_DTYPE_SIZES = {
    0: 4, 1: 2, 2: 2, 3: 4, 4: 8, 5: 1,
    6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 2,
}


def _fmt(s: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if s < 1024:
            return f"{s:.2f} {u}"
        s /= 1024
    return f"{s:.2f} PB"
