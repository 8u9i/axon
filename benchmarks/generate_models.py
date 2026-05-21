#!/usr/bin/env python3
"""Generate test .axon and .safetensors files for benchmarking."""

import os
import json
import struct
import hashlib
import numpy as np
import safetensors
import safetensors.numpy

def generate_axon(filename, tensors_dict, model_name="bench-test"):
    """Write a valid .axon v1.0 file from a dict of {name: np.ndarray}."""
    AXON_MAGIC = b"\x89.AXON\r\n\x1a\n"
    VERSION = 1

    # Build manifest
    manifest_data = {
        "model_name": model_name,
        "architecture": "benchmark",
        "quantization": "none",
        "checkpoint_format": {"type": "standard"},
        "framework": {"name": "benchmark", "version": "0.1.0"},
    }
    manifest_json = json.dumps(manifest_data, separators=(",", ":")).encode("utf-8")

    # Build tensor descriptors and data
    tensor_names = list(tensors_dict.keys())
    dtype_map = {
        np.float32: 0,
        np.float16: 1,
        np.float64: 7,
        np.int32: 4,
        np.int64: 5,
        np.uint8: 8,
        np.int8: 9,
    }

    tdt_entries = []
    tensor_data = b""

    for name in tensor_names:
        arr = tensors_dict[name]
        arr_contig = np.ascontiguousarray(arr)
        dtype_code = dtype_map.get(arr.dtype.type, 0)
        shape = list(arr.shape)

        name_bytes = name.encode("utf-8")
        tdt_entry = bytearray(192)
        off = 0

        name_buf = name_bytes.ljust(128, b"\x00")[:128]
        tdt_entry[off:off + 128] = name_buf
        off = 128

        tdt_entry[off:off + 1] = dtype_code.to_bytes(1, "little")
        off = 129

        ndim = len(shape)
        tdt_entry[off:off + 1] = ndim.to_bytes(1, "little")
        off = 130

        for dim in shape:
            tdt_entry[off:off + 8] = dim.to_bytes(8, "little")
            off += 8
        # Pad remaining shape slots with 0
        while off < 192:
            tdt_entry[off] = 0
            off += 1

        tdt_entries.append(bytes(tdt_entry))

    # Calculate offset for aligned data
    header_size = 64
    manifest_padded_size = ((len(manifest_json) + 63) // 64) * 64
    tdt_total_size = len(tdt_entries) * 192
    tdt_padded_size = ((tdt_total_size + 63) // 64) * 64
    data_offset = header_size + manifest_padded_size + tdt_padded_size

    # Build tensor data with correct offsets
    all_tdt_bytes = b""
    all_tensor_data = b""
    current_data_pos = data_offset

    for name in tensor_names:
        arr = np.ascontiguousarray(tensors_dict[name])
        name_bytes = name.encode("utf-8")
        dtype_code = dtype_map.get(arr.dtype.type, 0)
        shape = list(arr.shape)

        tdt_entry = bytearray(192)
        name_buf = name_bytes.ljust(128, b"\x00")[:128]
        tdt_entry[0:128] = name_buf
        tdt_entry[128] = dtype_code
        tdt_entry[129] = len(shape)
        off = 130
        for dim in shape:
            struct.pack_into("<Q", tdt_entry, off, dim)
            off += 8

        data_size = arr.nbytes
        struct.pack_into("<Q", tdt_entry, 136, current_data_pos)
        struct.pack_into("<Q", tdt_entry, 144, data_size)

        all_tdt_bytes += bytes(tdt_entry)
        all_tensor_data += arr.tobytes()
        current_data_pos += data_size

    tdt_padded = all_tdt_bytes + b"\x00" * (tdt_padded_size - len(all_tdt_bytes))
    manifest_padded = manifest_json + b"\x00" * (manifest_padded_size - len(manifest_json))

    # Header
    header = bytearray(64)
    header[0:10] = AXON_MAGIC
    struct.pack_into("<I", header, 10, VERSION)
    struct.pack_into("<Q", header, 16, len(all_tensor_data))  # data_size
    struct.pack_into("<Q", header, 24, len(tensor_names))     # tensor_count
    struct.pack_into("<Q", header, 32, header_size)            # manifest_offset
    struct.pack_into("<Q", header, 40, len(manifest_json))     # manifest_size
    struct.pack_into("<Q", header, 48, data_offset)            # tdt_start
    struct.pack_into("<Q", header, 56, len(tensor_names))      # tdt_count

    manifest_checksum = hashlib.xxh128(manifest_padded).digest()
    tdt_checksum = hashlib.xxh128(tdt_padded).digest()
    header[64-32:64-16] = manifest_checksum[:16]
    header[64-16:64] = tdt_checksum[:16]

    with open(filename, "wb") as f:
        f.write(bytes(header))
        f.write(manifest_padded)
        f.write(tdt_padded)
        f.write(all_tensor_data)

    return filename


def generate_safetensors(filename, tensors_dict):
    """Write a safetensors file."""
    safetensors.numpy.save_file(tensors_dict, filename)
    return filename


def create_model_configs():
    """Define the benchmark model configurations."""
    rng = np.random.RandomState(42)

    configs = []

    # Small: 10 tensors, ~10MB
    tensors_small = {}
    for i in range(10):
        shape = rng.choice([[64, 1024], [256, 256], [512, 512], [1024, 128], [4096, 64]])
        dtype = rng.choice([np.float16, np.float32])
        tensors_small[f"layer_{i}.weight"] = rng.randn(*shape).astype(dtype)
    configs.append(("model_10tensors_10mb", tensors_small))

    # Medium: 100 tensors, ~100MB
    tensors_medium = {}
    for i in range(100):
        shape = rng.choice([[128, 768], [768, 768], [768, 3072], [3072, 768], [1024, 1024]])
        dtype = rng.choice([np.float16, np.float32])
        tensors_medium[f"layer_{i}.weight"] = rng.randn(*shape).astype(dtype)
    configs.append(("model_100tensors_100mb", tensors_medium))

    # Large: 1000 tensors, ~1GB (F16 only to keep size manageable)
    tensors_large = {}
    # Use ~900KB per tensor on average
    for i in range(1000):
        shape = rng.choice([[256, 1024], [1024, 1024], [512, 2048]])
        tensors_large[f"layer_{i:04d}.weight"] = rng.randn(*shape).astype(np.float16)
    configs.append(("model_1000tensors_1gb", tensors_large))

    return configs


def main():
    os.makedirs("results", exist_ok=True)

    configs = create_model_configs()

    metadata = {
        "cpu": os.popen("lscpu 2>/dev/null | grep 'Model name' | head -1 || sysctl -n machdep.cpu.brand_string 2>/dev/null").read().strip() or "unknown",
        "ram": os.popen("free -h 2>/dev/null | grep Mem | awk '{print $2}' || vm_stat 2>/dev/null").read().strip() or "unknown",
        "os": os.popen("uname -srm 2>/dev/null").read().strip() or "unknown",
        "date": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
    }

    for name, tensors in configs:
        total_bytes = sum(arr.nbytes for arr in tensors.values())
        tb = total_bytes / (1024 ** 3) if total_bytes >= 1024 ** 3 else total_bytes / (1024 ** 2)

        print(f"\n{'='*60}")
        print(f"Generating: {name}")
        print(f"  Tensors: {len(tensors)}")
        print(f"  Total data: {total_bytes / (1024**2):.1f} MB ({total_bytes / (1024**3):.2f} GB)")
        print(f"  DTypes: {set(arr.dtype.name for arr in tensors.values())}")

        # Generate .axon
        axon_path = f"results/{name}.axon"
        generate_axon(axon_path, tensors)
        axon_size = os.path.getsize(axon_path) / (1024 ** 2)
        print(f"  .axon: {axon_path} ({axon_size:.1f} MB)")

        # Generate .safetensors
        st_path = f"results/{name}.safetensors"
        generate_safetensors(st_path, tensors)
        st_size = os.path.getsize(st_path) / (1024 ** 2)
        print(f"  .safetensors: {st_path} ({st_size:.1f} MB)")

    # Write metadata
    with open("results/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to results/metadata.json")
    print(f"Run benchmarks with: python bench_safetensors.py results/*.safetensors")


if __name__ == "__main__":
    main()
