# Paging Design (Experimental)

## Purpose

Enable tensor access for models whose total weight size exceeds available
RAM. Pages are loaded on demand from the backing store, evicted when
cold, and prefetched based on access patterns.

## Architecture (Future Vision)

```
SSD stores full model weights
    ↓ TensorPager trait
Page cache (configurable RAM budget)
    ↓ LRU eviction
Application code (only sees hot pages)
```

## TensorPager Trait

```rust
pub trait TensorPager {
    /// Get a page of tensor data.
    fn get_page(&self, tensor: &str, byte_offset: usize, len: usize) -> AxonResult<&[u8]>;

    /// Hint to prefetch a page range.
    fn prefetch(&self, tensor: &str, byte_offset: usize, len: usize) -> AxonResult<()>;

    /// Hint to evict a tensor's pages.
    fn evict(&self, tensor: &str) -> AxonResult<()>;
}
```

## Pluggable Backends

- **MmapStore** (current default): maps the full file, OS manages pages
- **NetworkPager**: loads pages from remote storage (S3, etc.)
- **CompressedPager**: decompresses on access, caches in RAM
- **ShardedPager**: pages from distributed shards

## Prefetch Strategy

- **Sequential**: prefetch N pages ahead during sequential tensor scans
- **Layer-aware**: prefetch entire layer when the first tensor in a
  layer is accessed (common in transformer inference)
- **ML-guided** (future): predict next access pattern from runtime stats

## Current PagedRuntime

A working `PagedRuntime` exists with:
- Fixed page size (default 4MB)
- LRU page cache with configurable page count
- Layer-aware prefetching
- `PagingStats` for hit/miss/fault tracking

## Current Status

**Experimental.** The `PagedRuntime` works for models that fit in the
virtual address space. The `TensorPager` trait is the extension point
for future implementations that support models larger than virtual
address space.

## Limitations

- Not yet validated against production-scale models
- No adaptive page sizing
- Prefetching is sequential only (no layer-aware predictive loading)
- No SSD-specific optimizations (queue depth, I/O scheduling)

## Future Work

- Prefetch strategy based on layer detection (transformer models)
- Adaptive page sizing based on access patterns
- SSD queue management for optimal I/O throughput
- Integration with inference runtimes
