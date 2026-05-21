# Zero-Copy Tensor Views

## Purpose

Axon provides borrowed `&[u8]` tensor views directly from memory-mapped
files. No allocation, no copying — the bytes live in the OS page cache
and the runtime exposes them through Rust's borrow semantics.

## Architecture

```
File on disk
    ↓ mmap (MAP_PRIVATE | PROT_READ)
OS page cache (RAM, page-faulted on demand)
    ↓ &[u8] borrow
Application code
```

The `MmapStore` holds an `mmap2::Mmap` handle. `raw_slice()` returns a
`&[u8]` whose lifetime is tied to the store itself. The runtime wraps
this with tensor lookup and validation.

## Public API

```rust
// Zero-copy — no allocation, no copying
pub fn tensor_view(&self, name: &str) -> AxonResult<&[u8]>;

// Zero-copy byte range
pub fn tensor_byte_view(&self, name: &str, range: Range<usize>) -> AxonResult<&[u8]>;

// Zero-copy row range (shape-aware)
pub fn tensor_rows(&self, name: &str, start_row: usize, end_row: usize) -> AxonResult<&[u8]>;
```

## Compatibility Layer

```rust
// Owned copy — outlives the runtime
pub fn tensor(&self, name: &str) -> AxonResult<Vec<u8>>;
```

## Lifetime Model

```rust
let rt = AxonRuntime::open("model.axon")?;
let view: &[u8] = rt.tensor_view("weight")?;
// view is valid as long as 'rt' exists (Rust's borrow checker enforces this)
```

The borrow checker guarantees that any `&[u8]` returned by the runtime
cannot outlive the `AxonRuntime` instance.

## Current Status

**Implemented.** All three zero-copy APIs are live and tested. The OS
manages page residency — accessed bytes are faulted in from disk on
demand.

## Limitations

- The entire file is mmap'd at open time (virtual address space).
  Very large models (e.g., >1TB) may exceed virtual address space on
  32-bit systems. 64-bit systems have ample room.
- Only contiguous ranges are supported. Non-contiguous tensor access
  (e.g., every other row) requires multiple calls.

## Future Work

- `TensorPager` trait will allow pluggable page sources for models
  larger than virtual address space.
- Prefetching support: `prefetch()` hint to warm up the page cache.
