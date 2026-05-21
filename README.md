<h1 align="center">
  <img src="https://img.shields.io/badge/🧬-AXON-8A2BE2" height="40" alt="AXON"><br>
  Adaptive eXecutable Object Notation
</h1>

<p align="center">
  <strong>A runtime-first model weight container for memory-limited inference.</strong><br>
  Instant loading · Memory-mapped tensor access · SSD-backed execution · LoRA side-loading<br>
  Mixed precision · 64-byte aligned · XXH3 checksums · SafeTensors import
</p>

<p align="center">
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  <a href="LICENSE-APACHE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
</p>

---

**.axon** is a binary model-weight container and runtime loader for AI models. It helps
memory-limited machines — laptops, edge devices, home AI servers — load and run large
models more efficiently by using memory mapping, partial tensor loading, and
SSD-backed caching.

## What Axon Does

- **Fast model opening** — parse header + manifest + tensor index in ~30µs, regardless
  of file size
- **Memory-mapped tensor access** — tensor bytes are faulted in from disk on demand,
  not eagerly loaded
- **Partial tensor loading** — load only the rows or byte range you need
- **SSD-backed execution** — keep model weights on SSD, cache only active tensors in RAM
- **LoRA adapter side-loading** — fast adapter switching without loading full models
- **Tensor cache management** — LRU eviction, pinning, memory budget control
- **Sharded model support** — models split across multiple files

## What Axon Is Not

Axon is **not** a training accelerator. Training speed depends on GPU compute, memory
bandwidth, optimizer operations, and matrix multiplication. Axon focuses on the
**storage and loading side** of model execution:

- Faster startup
- Lower memory overhead
- Partial tensor access
- SSD-backed model execution
- LoRA adapter switching
- Large-model usage on memory-limited machines

## Quick Start

```bash
# Install via Rust
cargo build --release

# Create a test model with 17 tensors
./target/release/axon create --model "MyModel-7B" --architecture llama model.axon

# Inspect it
./target/release/axon inspect model.axon

# Open with the runtime (zero-copy mmap, no tensor data loaded)
./target/release/axon runtime open model.axon
```

```rust
use axon_runtime::AxonRuntime;

let rt = AxonRuntime::open("model.axon")?;
println!("Model: {} ({} tensors)", rt.model_name(), rt.tensor_count());

// Tensor data is loaded lazily from the mmap — only the bytes you
// touch are faulted in from disk
let data = rt.tensor("emb_weight")?;
```

## Performance

Benchmarked on a ~100MB synthetic model (100 tensors, 1MB each):

| Operation | Time |
|---|---|
| Open (parse metadata) | **~29µs** |
| First tensor access | **~183ns** (offset math, OS handles page faults) |
| Sequential access (100 tensors) | **~498µs** |
| Partial load (4KB of 1MB tensor) | **~1.15µs** |
| Full load (1MB tensor) | **~144µs** |

**Key insight:** The runtime does not load tensor data during `open()`. Only the
header (64 bytes), manifest (a few KB), and tensor descriptors (192 bytes each) are
parsed. Individual tensor bytes are faulted in from disk by the OS on first access.

## Runtime Architecture

Axon has two layers:

| Crate | Purpose | Memory model |
|---|---|---|
| `core/` | Format library: parse, write, validate, convert | Loads into `Vec<u8>` (safe, simple) |
| `runtime/` | Execution layer: mmap, cache, partial load, LoRA | Borrows from mmap (zero-copy, lazy) |

The runtime is the recommended path for inference. The core format library is the
stable base used by the CLI, FFI, and Python bindings.

See **[docs/runtime-architecture.md](docs/runtime-architecture.md)** for the full design.

## Project Structure

```
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

```
axon create      Create a synthetic .axon file for testing
axon inspect     Show file structure and tensor list
axon validate    Verify structure and checksums
axon list        List all tensors
axon extract     Extract a single tensor by name
axon unpack      Extract all tensors to .npy or .bin files
axon pack        Pack tensors from a manifest + data directory
axon convert     Export manifest as JSON
axon bench       Benchmark load/index performance
axon runtime     Runtime subcommands (open, tensor, stats)
```

## Documentation

- **[docs/spec.md](docs/spec.md)** — Binary format specification
- **[docs/runtime-architecture.md](docs/runtime-architecture.md)** — Runtime design
- **[docs/usage.md](docs/usage.md)** — CLI, Python, and C FFI usage

## License

Dual-licensed under [Apache 2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT).
