# Axon Runtime Architecture

**Version:** 0.1 (Phase 1)
**Status:** Draft

---

## 1. Why a Separate Runtime Layer

The `axon-core` crate implements the binary format: reading, writing, validating, and
serializing `.axon` files. It is a **format library**. It loads the whole file into
memory as `Vec<u8>` because that is the safest and simplest contract for a format
parser.

The `axon-runtime` crate is an **execution layer** built on top of the format. It
addresses a different set of requirements:

| Concern | `axon-core` (format) | `axon-runtime` (execution) |
|---|---|---|
| Memory model | Loads entire file into `Vec<u8>` | Borrows bytes from an mmap, never copies |
| Tensor access | Returns `&[u8]` slices into owned buffer | Returns borrowed views into the mmap |
| File size limit | Bounded by available RAM | Bounded by virtual address space (mmap) |
| Caching | None | LRU cache with configurable memory budget |
| Partial loading | None | Slice rows/columns without loading full tensor |
| LoRA/adapter support | Manifest has patch metadata only | Runtime side-loading with zero-copy shadowing |
| Execution context | Single-threaded parsing | Designed for concurrent access patterns |

The two crates have different stability guarantees. `axon-core` follows semver on
the format spec. `axon-runtime` may iterate faster as we discover real-world usage
patterns.

---

## 2. Crate Structure

```
crates/axon-runtime/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public re-exports, top-level API
│   ├── runtime.rs              # AxonRuntime — the main entry point
│   ├── mmap_store.rs           # MmapStore: zero-copy mmap abstraction
│   ├── tensor_cache.rs         # TensorCache: LRU eviction + pinning
│   ├── slice.rs                # Tensor slicing and partial loading
│   ├── lora.rs                 # PatchRuntime: LoRA side-loading (Phase 4)
│   ├── paging.rs               # SSD-backed tensor paging (Phase 5, experimental)
│   └── stats.rs                # Runtime statistics and instrumentation
```

### 2.1 Dependency Graph

```
axon-runtime
  ├── axon-core      (format parsing, DType, TensorDescriptor)
  ├── memmap2        (mmap abstraction)
  └── lru            (LRU cache for tensors)

axon-cli
  └── axon-runtime   (runtime subcommand)

python/axon
  └── axon-runtime   (FFI or ctypes to runtime)
```

`axon-runtime` depends on `axon-core` for:
- `AxonHeader` — parse header from mmap
- `TensorDescriptor` — parse TDT entries from mmap
- `Manifest` — parse manifest JSON from mmap
- `DType` — type information for slicing calculations
- `AxonError` / `AxonResult` — consistent error types

It does **not** depend on `AxonFile`, `MappedAxonFile`, or `AxonBuilder` — those
are format-layer types that copy data. The runtime has its own mmap path.

---

## 3. Ownership and Lifetime Model

The central design challenge: **tensor views must outlive the runtime, but borrow
from the mmap.**

### 3.1 Current `core` approach (copying)

```
AxonFile.from_bytes(data: Vec<u8>)
  └── self.data: Vec<u8>          ← owns the bytes
       └── tensor_data() -> &[u8] ← borrows from self.data
```

This works because `AxonFile` owns the data. But it requires loading everything
into RAM.

### 3.2 Runtime approach (true mmap, owned)

The runtime owns the mmap handle. Tensor views are scoped slices — not `&[u8]`
references into the mmap (which would tie lifetimes), but **owned** byte ranges
that can be independently cached, paged, or copied.

```rust
pub struct AxonRuntime {
    store: MmapStore,          // owns the mmap handle
    header: AxonHeader,        // parsed copy (tiny, 64 bytes)
    manifest: Manifest,        // parsed copy (few KB)
    descriptors: Vec<TensorDescriptor>,  // parsed copy (192 bytes per tensor)
    cache: TensorCache,        // optional LRU cache
}
```

Key lifetime rule: **the runtime owns everything that has a lifetime.** The mmap
is owned by `MmapStore`, which is owned by `AxonRuntime`. Users never hold
references into the mmap directly — they receive `Arc<Vec<u8>>` from the cache,
or `Vec<u8>` from eager loads.

This means:

- **Zero-copy reads**: When a tensor is served from the mmap without caching, the
  runtime copies the bytes into a `Vec<u8>` and returns it. This seems wasteful,
  but it is the only safe API for a library: the user can drop the runtime and
  still hold the tensor data.
- **Zero-copy cache hits**: When a tensor is in the cache, the runtime returns
  an `Arc<Vec<u8>>` — no copy, just a reference count increment.
- **Future optimization**: Once we add lifetime-scoped access (`RuntimeSession`),
  we can return true borrowed `&[u8]` slices for zero-copy read-only access
  within a scoped context. This is Phase 2 or 3 work.

### 3.3 Design decision rationale

| Option | Pros | Cons |
|---|---|---|
| Borrowed `&[u8]` from mmap | True zero-copy for all reads | Ties tensor lifetime to runtime lifetime; complex borrow checker gymnastics |
| `Arc<Vec<u8>>` from cache | Safe, simple, outlives runtime | Copy on first access (mmap → Vec); amortized by cache |
| `Cow<[u8]>` | Either borrowed or owned | Forces all consumers to handle both variants; API complexity |
| Scoped session (future) | Zero-copy within session | Two-tier API; more surface area |

**Chosen: `Arc<Vec<u8>>` for Phase 1.** Copy-on-first-access is transparent to
users, safe, and straightforward. The cache eliminates the copy cost for hot
tensors. Benchmarks will show whether the copy is a bottleneck — if it is, we
add scoped sessions later.

---

## 4. Mmap Strategy

### 4.1 File layout (from spec)

```
Offset 0:     AxonHeader (64 bytes)
Offset 64:    HOT_START padding to 4096
Offset 4096:  Manifest JSON (variable size)
              padding to 64
              TensorDescriptor Table (192 bytes × N tensors)
              padding to 64
Offset P:     Tensor payload 0 (64-byte aligned)
              Tensor payload 1
              ...
              Tensor payload N-1
```

### 4.2 Mapping strategy

```
AxonRuntime::open(path)
  1. Open file, get length
  2. mmap the entire file (MAP_PRIVATE, PROT_READ)
  3. Parse header from bytes [0..64)
  4. Parse manifest from [manifest_offset..manifest_offset+manifest_size)
  5. Parse TDT from [tdt_start..tdt_start + tensor_count * 192)
  6. Store MmapStore, header, manifest, descriptors
  7. Runtime is ready — no tensor data has been touched
```

**Key properties:**

- The mmap is `MAP_PRIVATE` — writes (if any) are copy-on-write, never written
  back to the file.
- The mmap is `PROT_READ` only — no accidental mutation.
- The OS handles page faults. Touching a tensor's byte range for the first time
  causes a page fault, the OS loads the page from the file cache or SSD, and
  subsequent accesses are RAM hits.
- A 10GB file with a 4KB hot start will fault in only ~8 pages on open (header +
  manifest + TDT). The remaining 2.6 million pages are untouched until tensor
  access.

### 4.3 MmapStore

```rust
/// Owns the mmap and provides safe byte-range access.
pub struct MmapStore {
    mmap: Mmap,
    len: u64,
}

impl MmapStore {
    /// Open a file and mmap it.
    pub fn open(path: &Path) -> AxonResult<Self>;

    /// Read a byte range from the mmap. Returns owned bytes.
    /// This is the primary access method — it triggers a page fault
    /// on first access to the byte range.
    pub fn read_bytes(&self, offset: u64, size: u64) -> AxonResult<Vec<u8>>;

    /// Get a reference to raw mmap bytes. Must not outlive the store.
    pub(crate) fn raw_slice(&self, offset: u64, size: u64) -> Option<&[u8]>;

    /// The total file size.
    pub fn len(&self) -> u64;
}
```

---

## 5. Cache Design (Phase 2)

### 5.1 TensorCache

```rust
pub struct TensorCache {
    inner: LruCache<String, Arc<Vec<u8>>>,
    memory_budget: usize,           // Max bytes to cache
    current_usage: Arc<AtomicUsize>, // Tracked current usage
    pinned: HashSet<String>,        // Never evict these
    stats: CacheStats,
}

impl TensorCache {
    pub fn new(memory_budget: usize) -> Self;
    pub fn get(&mut self, name: &str) -> Option<Arc<Vec<u8>>>;
    pub fn put(&mut self, name: String, data: Vec<u8>) -> Arc<Vec<u8>>;
    pub fn pin(&mut self, name: &str);
    pub fn unpin(&mut self, name: &str);
    pub fn stats(&self) -> CacheStats;
    pub fn evict(&mut self, name: &str);
}
```

**Eviction policy:** LRU (Least Recently Used). When `put` would exceed the
memory budget, evict the least recently used unpinned tensor(s) until the budget
is satisfied.

**Pinned tensors:** Frequently accessed tensors (embedding tables, LM heads,
layer norms) can be pinned to prevent eviction. Pinned tensors count toward
the memory budget.

**Thread safety:** Phase 1 uses `&mut self` (single-threaded). Phase 2 or 3 can
add `Mutex` or `RwLock` for concurrent access.

### 5.2 Integration with AxonRuntime

```rust
impl AxonRuntime {
    /// Create a runtime with an LRU cache of the given size.
    pub fn with_cache(path: &Path, cache_size: usize) -> AxonResult<Self>;

    /// Get a tensor. Checks cache first, falls back to mmap.
    pub fn tensor(&mut self, name: &str) -> AxonResult<Arc<Vec<u8>>>;

    /// Pin a tensor in the cache.
    pub fn pin(&mut self, name: &str);

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats;
}
```

---

## 6. Partial Tensor Loading (Phase 2)

### 6.1 Slice API

```rust
impl TensorSliceSpec {
    /// Slice by byte range (raw)
    pub fn byte_range(offset: usize, size: usize) -> Self;

    /// Slice by rows (for 2D row-major tensors)
    pub fn rows(row_start: usize, row_count: usize) -> Self;

    /// Slice by rows and columns (for 2D row-major tensors)
    pub fn row_col(row_start: usize, row_count: usize,
                   col_start: usize, col_count: usize) -> Self;
}

impl AxonRuntime {
    /// Load a portion of a tensor. Does not cache the result
    /// (the user gets the bytes they asked for, not the full tensor).
    pub fn tensor_slice(&self, name: &str, spec: TensorSliceSpec) -> AxonResult<Vec<u8>>;

    /// Load a contiguous byte range from a tensor.
    /// This avoids pulling the entire tensor into memory.
    pub fn tensor_bytes(&self, name: &str, offset: u64, size: u64) -> AxonResult<Vec<u8>>;
}
```

### 6.2 Page-fault semantics

When `tensor_slice` is called, the runtime:

1. Looks up the tensor descriptor to get `data_offset` and `data_size`
2. Validates the slice bounds against the tensor shape and dtype
3. Reads only the requested byte range from the mmap
4. Returns the bytes without touching any other part of the tensor

The OS only pages in the SSD blocks that cover the requested range. If the
tensor is 500MB and the slice is 4KB, exactly 4KB is read from disk (plus
one filesystem block).

---

## 7. Benchmark Methodology

### 7.1 Benchmark framework

The benchmark suite lives in `benches/runtime/` and uses Criterion.rs for
precise measurement.

### 7.2 Benchmark scenarios

| Benchmark | What it measures |
|---|---|
| `open_time` | Time to parse header + manifest + TDT from an mmap'd file |
| `first_tensor_access` | Time to access the first tensor after open |
| `random_tensor_access` | Time to access 10 randomly selected tensors |
| `sequential_tensor_access` | Time to iterate all tensors in order |
| `peak_memory` | Peak RSS during tensor access patterns |
| `slice_vs_full` | Memory/time comparison: load full tensor vs load a slice |

### 7.3 Comparison targets

- **.axon (runtime)**: Using the new `AxonRuntime`
- **.safetensors**: Using the `safetensors` Rust crate (or Python)
- **PyTorch .pt**: Using `torch.load` (Python only)

### 7.4 Output format

Each benchmark run produces a Markdown table:

```text
| Format | Open | First Tensor | 10 Random | Peak RSS |
|--------|------|--------------|-----------|----------|
| .axon  | 12µs | 3µs          | 28µs      | 48MB     |
| .safetensors | 45ms | 12ms  | 95ms      | 2.1GB    |
| .pt    | 820ms | 820ms       | 950ms     | 2.2GB    |
```

### 7.5 Test data generation

Benchmarks use synthetic data generated by `AxonBuilder` (from core):

```bash
# Generate test files of various sizes
axon create --model "bench-100MB"  bench_100mb.axon
axon create --model "bench-1GB"    bench_1gb.axon
axon create --model "bench-10GB"   bench_10gb.axon
```

The `create` subcommand already generates a 1.1GB file with 17 tensors — we
extend it with size options.

---

## 8. Future Extension Points

### 8.1 Paging (Phase 5, experimental)

The paging system will allow a runtime model to have **more tensor data than
available RAM** by treating the mmap as a backing store and the cache as a
working set.

```rust
// Future API sketch (not implemented yet)
pub struct PagedRuntime {
    runtime: AxonRuntime,
    page_size: u64,         // e.g., 4MB pages
    policy: PagePolicy,      // LRU, FIFO, or learned
}

impl PagedRuntime {
    /// Tensor access may trigger page-in from SSD if not in cache.
    pub fn tensor_paged(&mut self, name: &str) -> AxonResult<Arc<Vec<u8>>>;

    /// Prefetch a set of tensors into the cache.
    pub fn prefetch(&mut self, names: &[&str]);

    /// Release a tensor from the cache (hint to evict).
    pub fn release(&mut self, name: &str);
}
```

The abstraction that makes paging possible: **tensor access always goes through
the cache.** Today the cache is optional. Tomorrow, when paging is enabled, the
cache becomes mandatory — tensor data that isn't in the cache is on SSD, and
accessing it pages it in.

### 8.2 Predictive prefetching

Future work can add:

- Layer-aware prefetching: if the model is a transformer, predict which layer's
  tensors will be needed next and load them ahead of time.
- Access pattern tracking: record which tensors are accessed together and
  prefetch the group.
- ML-driven prefetching: train a lightweight model to predict tensor access
  patterns based on input sequence length, batch size, or other signals.

### 8.3 Scoped zero-copy access

```rust
// Future API sketch
pub struct RuntimeSession<'a> {
    runtime: &'a AxonRuntime,
}

impl<'a> RuntimeSession<'a> {
    /// Returns a zero-copy view into the mmap.
    /// The view is valid only while the session exists.
    pub fn tensor_view(&self, name: &str) -> Option<&'a [u8]>;
}
```

This would allow true zero-copy access within a scoped context, eliminating the
copy even on first access. It requires the user to hold the session (and thus
the runtime) alive for the duration of access.

---

## 9. Public API Surface (Phase 1)

### 9.1 Rust

```rust
// Open a file and parse metadata (no tensor data touched)
let runtime = AxonRuntime::open("model.axon")?;

// Get tensor data (first access triggers page fault + copy)
let data: Arc<Vec<u8>> = runtime.tensor("layer_0_q")?;

// Get tensor metadata without loading data
let info: TensorInfo = runtime.tensor_info("layer_0_q")?;

// List all tensors
let tensors: Vec<TensorInfo> = runtime.tensors();

// Runtime statistics
let stats = runtime.stats();
```

### 9.2 Python (Phase 3)

```python
import axon

# Open with runtime (zero-copy mmap)
model = axon.runtime.open("model.axon")

# Get tensor
data = model.tensor("layer_0_q")

# Metadata (no data loaded)
info = model.tensor_info("layer_0_q")
print(info.name, info.dtype, info.shape, info.size_bytes)

# List tensors
for t in model.tensors():
    print(t)
```

### 9.3 CLI (Phase 3)

```bash
# Open and show statistics
axon runtime open model.axon

# Access a specific tensor
axon runtime tensor model.axon layer_0_q

# Show cache hit rate (if cache enabled)
axon runtime stats model.axon
```
