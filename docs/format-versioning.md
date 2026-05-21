# Axon Format Versioning Policy

Axon files carry a format version in the 64-byte `AxonHeader`. The current
format version is `1`.

## Compatibility Rules

- Readers must reject files whose header magic is not `AXON`.
- Readers must reject unsupported format versions with a typed error.
- Version `1` readers may ignore manifest metadata keys they do not understand.
- Tensor layout fields in the descriptor table are authoritative. Manifest
  metadata must not override descriptor offsets, sizes, dtype, rank, shape, or
  checksums.
- Reserved descriptor/header fields must be written as zero unless a future
  version defines them.
- Unknown flag bits must be treated as unsupported unless the reader explicitly
  documents that it can ignore them safely.

## Version Changes

Use a new format version when a change affects binary parsing or tensor payload
interpretation.

Examples that require a new format version:

- changing `AxonHeader` size or field order
- changing `TensorDescriptor` size or field order
- changing dtype numeric codes
- changing endianness
- changing whether descriptor offsets are absolute or relative
- adding mandatory compression, encryption, or sharding semantics

Examples that do not require a new format version:

- adding optional manifest metadata
- adding new CLI commands
- improving validation or diagnostics
- adding bindings for another language
- adding optional tooling around existing v1 files

## Writer Rules

Writers that produce v1 files must:

- write little-endian integer fields
- write a 64-byte aligned header and tensor descriptor table
- write 64-byte aligned tensor payload offsets
- write tensor descriptors as the source of truth for tensor layout
- compute tensor checksums when `HAS_CHECKSUMS` is set

## Reader Rules

Readers that consume v1 files must:

- validate magic and version before parsing later regions
- bounds-check manifest, descriptor table, and tensor payload offsets before
  slicing or mmap access
- reject invalid dtype codes and impossible ranks
- fail cleanly on truncated files or malformed manifests
- expose clear errors at CLI, FFI, Python, and Rust API boundaries
