# .axon Specification v1.0

See [format-versioning.md](format-versioning.md) for reader/writer
compatibility rules and the policy for future format versions.

**Adaptive eXecutable Object Notation** — a binary-first, hardware-aligned container format for machine learning model weights.

- **Zero-copy loading**: 64-byte aligned for direct mmap into GPU/NPU address space
- **Mixed precision**: Supports Int4, FP8, BF16, FP16, FP32, and integer types in a single file
- **Checksummed**: Every tensor has an XXH3-64 checksum embedded in its descriptor
- **Self-describing**: Embedded JSON-LD manifest with full metadata
- **Shard-ready**: Logical stripping for multi-device distribution without re-sharding
- **Patchable**: Side-loading headers for LoRA/adapters without modifying the base file

## 1. File Layout

```
┌─────────────────────────────────────────────┐
│  AxonHeader (64 bytes, 64-byte aligned)      │  ← Offset 0
├─────────────────────────────────────────────┤
│  HOT_START padding (4032 bytes)              │  ← Total: 4KB
├─────────────────────────────────────────────┤
│  Manifest (variable, JSON-LD)                │  ← manifest_offset
├─────────────────────────────────────────────┤
│  Padding to 64-byte boundary                  │
├─────────────────────────────────────────────┤
│  TensorDescriptor Table (192 bytes each)     │  ← TDT start
├─────────────────────────────────────────────┤
│  Padding to 64-byte boundary                  │
├─────────────────────────────────────────────┤
│  Tensor Payload 0 (64-byte aligned)          │  ← payload_offset
│  Tensor Payload 1 (64-byte aligned)          │
│  ...                                         │
│  Tensor Payload N (64-byte aligned)          │
└─────────────────────────────────────────────┘
```

## 2. Header (64 bytes)

| Offset | Size | Type     | Field            | Description                               |
|--------|------|----------|------------------|-------------------------------------------|
| 0      | 4    | char[4]  | magic            | Magic bytes: `AXON` (0x41 0x58 0x4F 0x4E)|
| 4      | 4    | u32      | version          | Format version (currently 1)              |
| 8      | 8    | u64      | manifest_offset  | Byte offset to the start of the manifest  |
| 16     | 8    | u64      | manifest_size    | Size of the manifest in bytes             |
| 24     | 8    | u64      | tensor_count     | Number of tensors in the file             |
| 32     | 8    | u64      | payload_offset   | Byte offset to the first tensor payload   |
| 40     | 8    | u64      | payload_size     | Total size of all tensor payloads in bytes|
| 48     | 8    | u64      | checksum         | XXH3-64 checksum of bytes [0..48)         |
| 56     | 8    | u64      | flags            | Bit flags (see below)                     |

### Flags

| Bit | Meaning     | Description                                |
|-----|-------------|--------------------------------------------|
| 0   | HAS_CHECKSUMS | Per-tensor XXH3 checksums are present    |
| 1   | SHARDED     | File is part of a multi-shard model        |
| 2   | INTERLEAVED | Weights use interleaved storage for multi-GPU |
| 3   | COMPRESSED  | Payload is compressed                     |
| 4   | ENCRYPTED   | Payload is encrypted                      |

## 3. Manifest (JSON-LD)

The manifest is a JSON-LD block that stores all model metadata. It is human-readable and self-describing.

```json
{
  "model": "MyModel-7B",
  "architecture": "llama",
  "hyperparameters": {
    "hidden_size": 4096,
    "num_layers": 32,
    "num_heads": 32
  },
  "tokenizer": {
    "model_type": "tiktoken",
    "vocab_size": 32000,
    "bos_token": "<s>",
    "eos_token": "</s>"
  },
  "context_length": 8192,
  "quantization": {
    "method": "gptq",
    "group_size": 128,
    "bits": 4
  },
  "patches": [
    {
      "name": "lora-fine-tune-v2",
      "file": "patches/lora-v2.axon",
      "description": "LoRA fine-tune with rank 64"
    }
  ]
}
```

**Note**: The manifest does not redundantly store tensor metadata; the `TensorDescriptor` table is the authoritative source for all tensor layout information.

## 4. Tensor Descriptor (192 bytes)

| Offset | Size | Type       | Field        | Description                               |
|--------|------|------------|--------------|-------------------------------------------|
| 0      | 64   | char[64]   | name         | Null-terminated tensor name               |
| 64     | 4    | u32        | dtype        | Data type code (see below)                |
| 68     | 4    | u32        | rank         | Number of dimensions (1-8)                |
| 72     | 64   | u64[8]     | shape        | Dimension sizes (only `rank` are valid)   |
| 136    | 8    | u64        | data_offset  | Absolute byte offset to raw tensor data   |
| 144    | 8    | u64        | data_size    | Size of tensor data in bytes              |
| 152    | 4    | u32        | affinity     | Memory affinity hint (see below)          |
| 156    | 4    | u32        | padding      | Reserved, must be 0                       |
| 160    | 8    | u64        | checksum     | XXH3-64 checksum of the tensor data       |

### DType Codes

| Code | Name      | Size | Description                    |
|------|-----------|------|--------------------------------|
| 0    | FP32      | 4B   | IEEE 754 single-precision      |
| 1    | FP16      | 2B   | IEEE 754 half-precision        |
| 2    | BF16      | 2B   | BFloat16 (truncated FP32)      |
| 3    | I32       | 4B   | Signed 32-bit integer          |
| 4    | I64       | 8B   | Signed 64-bit integer          |
| 5    | U8        | 1B   | Unsigned 8-bit integer        |
| 6    | Q4        | 1B*  | 4-bit quantized (packed)       |
| 7    | Q8        | 1B   | 8-bit quantized                |
| 8    | FP8_E4M3  | 1B   | FP8 4 exponent / 3 mantissa   |
| 9    | FP8_E5M2  | 1B   | FP8 5 exponent / 2 mantissa   |
| 10   | I8        | 1B   | Signed 8-bit integer           |
| 11   | I16       | 2B   | Signed 16-bit integer          |

*\* Q4 stores 2 values per byte*

### Affinity Hints

| Code | Name      | Description                           |
|------|-----------|---------------------------------------|
| 0    | Default   | Let the loader decide                 |
| 1    | HBM       | Place in High-Bandwidth Memory (GPU)  |
| 2    | SystemRAM | Place in system RAM (CPU)             |
| 3    | LLC       | Place in Last-Level Cache             |

## 5. Data Types

### Endianness

All multi-byte integers and floats are stored in **little-endian** byte order.

### Alignment

- Header: 64-byte aligned
- Manifest: no alignment requirement
- TensorDescriptor Table: 64-byte aligned
- Tensor payloads: 64-byte aligned (matching CPU cache line size)

### Checksums

- Header checksum: XXH3-64 over bytes [0..48) of the header
- Per-tensor checksum: XXH3-64 over the raw tensor data bytes
- A checksum value of 0 means "not computed/not verified"

## 6. Loading Protocol

1. Read first 64 bytes → parse header
2. Validate magic (`AXON`) and version (1)
3. Jump to `manifest_offset` → read `manifest_size` bytes → parse JSON manifest
4. Jump to TDT (after manifest, 64-byte aligned) → read `tensor_count × 192` bytes → parse tensor descriptors
5. Jump to `payload_offset` → access tensor data directly via offsets in descriptors

For zero-copy loading:
```c
// mmap the entire file
void* data = mmap(file, size, PROT_READ, MAP_PRIVATE, fd, 0);
// Cast the header directly
AxonHeader* header = (AxonHeader*)data;
// Access tensor data by offset
float* weights = (float*)((char*)data + descriptor->data_offset);
```

## 7. Side-Loading (Patches)

.axon supports a patching mechanism for LoRA and adapter weights:

1. A patch file is a standard .axon file with the same tensor names as the base
2. The loader checks the `patches` array in the manifest
3. For each patch tensor, the loader redirects lookups to the patch file's offset
4. This enables zero-copy LoRA inference without modifying the base file

## 8. Comparison with Other Formats

| Feature              | .json               | .safetensors        | .axon (this spec)      |
|----------------------|---------------------|---------------------|------------------------|
| Parsing speed        | Extremely slow      | Fast (binary)       | Instant (mmap)         |
| Metadata             | Flexible but bulky  | Limited             | Dynamic binary schemas |
| Tensor support       | None (Base64 hack)  | Excellent           | Native ND-array        |
| Mixed precision      | No                  | No                  | Yes (per-tensor)       |
| Multi-GPU sharding   | Manual              | Manual              | Built-in (flags)       |
| Checksums            | No                  | Yes                 | Yes (XXH3)             |
| LoRA patches         | No                  | No                  | Yes (side-loading)     |
| Hardware alignment   | No                  | No                  | Yes (64-byte)          |
| C FFI                | N/A                 | N/A                 | Yes (axon_ffi)         |
