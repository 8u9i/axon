<h1 align="center">
  <img src="https://img.shields.io/badge/AXON-8A2BE2" height="40" alt="AXON"><br>
  Adaptive eXecutable Object Notation
</h1>

<p align="center">
  <img src="assets/axon-diagram.jpeg" alt="Axon Architecture Diagram" width="800">
</p>

<p align="center">
  <strong>A runtime-first model-weight container for memory-limited inference.</strong><br>
  Instant loading &middot; Memory-mapped tensor access &middot; SSD-backed execution &middot; LoRA side-loading<br>
  Mixed precision &middot; 64-byte aligned &middot; XXH3 checksums &middot; SafeTensors import
</p>

<p align="center">
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  <a href="LICENSE-APACHE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
</p>

---

**.axon** is a binary model-weight container and runtime loader for AI models.
It helps memory-limited machines, such as laptops, edge devices, and home AI
servers, load large model weights with memory mapping, partial tensor access,
and SSD-backed caching.

## What Axon Does

- **Zero-copy tensor views via mmap**: `tensor_view()` returns a `&[u8]` slice
  directly into the memory-mapped file. No allocation, no copying. The OS pages
  tensor data from disk on first access.
- **Fast model opening**: parse the header, manifest, and tensor index in
  microseconds, regardless of file size.
- **Partial tensor loading**: load only the rows or byte range needed by the
  runtime.
- **SSD-backed execution**: keep weights on SSD and cache only active tensors in
  RAM.
- **LoRA adapter side-loading**: switch adapters without loading full model
  copies.
- **Tensor cache management**: LRU eviction, pinning, and memory budget control.
- **Sharded model support**: support models split across multiple files.

## What Axon Is Not

Axon is not a neural-network training framework and does not accelerate training
compute. Training speed still depends on GPU compute, memory bandwidth, matrix
multiplication kernels, and optimizer implementation.

Axon improves:

- model loading and startup latency
- memory usage through lazy mmap and partial access
- tensor access through zero-copy views
- runtime deployment through caching and patching
- inference startup latency on memory-constrained hardware

The primary Axon value proposition:

> Axon provides safe, mmap-backed, low-memory, partial-access tensor loading for
> runtime inference workloads.

Target environments:

- laptops
- mini PCs
- Jetson and edge AI devices
- Raspberry Pi-class systems
- local AI inference servers
- memory-constrained deployments

## Quick Start

```bash
# Build the Rust workspace
cargo build --release

# Create a synthetic test model with 17 tensors
./target/release/axon create --model "MyModel-7B" --architecture llama model.axon

# Inspect and validate it
./target/release/axon inspect model.axon
./target/release/axon validate model.axon

# Open with the runtime inspection path. No tensor bytes are loaded up front.
./target/release/axon runtime inspect model.axon
```

```rust
use axon_runtime::AxonRuntime;

let rt = AxonRuntime::open("model.axon")?;

// Zero-copy view: no allocation, no copying, direct mmap slice.
let view: &[u8] = rt.tensor_view("emb_weight")?;

// Shape-aware row slicing: maps only the requested rows.
let rows: &[u8] = rt.tensor_rows("emb_weight", 0, 128)?;

// Owned copy that can outlive the runtime.
let data: Vec<u8> = rt.tensor("emb_weight")?;
```

## Python Bindings

```bash
cargo build --release -p axon-ffi
pip install -e ./python
python examples/python_example.py model.axon
```

To build Python distribution artifacts:

```bash
python -m pip install build
python -m build ./python
```

```python
import axon

model = axon.load("model.axon")
print(model.summary())
tensor = model.tensor("emb_weight")
```

The Python package uses the native FFI library when available and keeps a
pure-Python fallback for basic loading workflows.

## C FFI

```c
#include "axon.h"

AxonHandle* handle = axon_open("model.axon");
if (!handle) {
    char err[256];
    axon_last_error(err, sizeof(err));
    return 1;
}

uint64_t count = axon_tensor_count(handle);
axon_close(handle);
```

Pointers returned by the FFI borrow from the mapped file and remain valid only
while the `AxonHandle` is open.

## Performance

Benchmarked on a synthetic model with 100 tensors of 1 MB each:

| Operation | Time |
|---|---:|
| Open (parse metadata) | ~29 us |
| First tensor access | ~183 ns |
| Sequential access (100 tensors) | ~498 us |
| Partial load (4 KB of 1 MB tensor) | ~1.15 us |
| Full load (1 MB tensor) | ~144 us |

Key insight: the runtime does not load tensor data during `open()`. Only the
header, manifest, and tensor descriptors are parsed. Individual tensor bytes are
faulted in from disk by the operating system on first access.

## Runtime Architecture

Axon has two layers:

| Crate | Purpose | Memory model |
|---|---|---|
| `core/` | Format library: parse, write, validate, convert | Loads into `Vec<u8>` |
| `runtime/` | Execution layer: mmap, cache, partial load, LoRA | Borrows from mmap |

The runtime is the recommended path for inference. The core format library is
the stable base used by the CLI, FFI, and Python bindings.

See [docs/runtime-architecture.md](docs/runtime-architecture.md) for the full
design.

## Project Structure

```text
axon/
├── core/              # Core format library (Rust)
├── runtime/           # SSD-backed lazy runtime (Rust)
├── cli/               # Command-line tool
├── ffi/               # C FFI shared library
├── python/            # Python package (ctypes + pure-Python fallback)
├── include/           # C header (axon.h)
├── docs/              # Spec and architecture docs
├── tests/             # Integration tests
└── examples/          # Usage examples
```

## CLI Reference

```text
axon create      Create a synthetic .axon file for testing
axon inspect     Show file structure and tensor list
axon validate    Verify structure and checksums
axon list        List all tensors
axon extract     Extract a single tensor by name
axon unpack      Extract all tensors to .npy or .bin files
axon pack        Pack tensors from a manifest and data directory
axon convert     Export manifest as JSON
axon bench       Benchmark load/index performance
axon import-gguf Import a GGUF model into .axon format
axon import-ollama Import a locally installed Ollama model into .axon format
axon runtime     Runtime subcommands: inspect, tensor, slice, stats, bench
```

## Ollama and GGUF

Ollama models are commonly stored as GGUF blobs. Axon can import GGUF v2/v3
files into `.axon` so you can inspect, validate, checksum, and partially access
their tensor bytes with Axon tooling:

```bash
axon import-gguf model.gguf --output model.axon
axon inspect model.axon
axon runtime tensor model.axon token_embd.weight
```

If Ollama already has the model installed, Axon can read Ollama's local
manifest/blob store directly without starting the Ollama server:

```bash
axon import-ollama gemma3:1b --output gemma3-1b.axon
axon inspect gemma3-1b.axon
axon validate gemma3-1b.axon
```

This importer preserves GGUF tensor byte ranges and metadata for Axon workflows.
It does not dequantize tensors or make `.axon` directly runnable by Ollama; use
Ollama's native GGUF import path when you want to run the model in Ollama.

## Documentation

- [docs/spec.md](docs/spec.md): binary format specification
- [docs/format-versioning.md](docs/format-versioning.md): compatibility and
  versioning policy
- [docs/runtime-architecture.md](docs/runtime-architecture.md): runtime design
- [docs/usage.md](docs/usage.md): CLI, Python, and C FFI usage
- [CONTRIBUTING.md](CONTRIBUTING.md): development workflow and quality gates
- [CHANGELOG.md](CHANGELOG.md): release history and unreleased changes

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT).
