//! # RuntimeStats
//!
//! Instrumentation counters for the runtime. Tracks tensor access patterns,
//! cache behavior, and I/O that can be used for diagnostics or prefetching
//! hints in later phases.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic instrumentation counters for the runtime.
///
/// All counters use relaxed ordering — we don't need strict consistency
/// for diagnostic metrics, just eventual visibility.
#[derive(Debug, Default)]
pub struct RuntimeStats {
    /// Number of tensor access calls made.
    pub(crate) tensor_accesses: AtomicU64,
    /// Number of bytes read from the mmap (cumulative).
    pub(crate) bytes_read: AtomicU64,
    /// Number of cache hits (when cache is enabled).
    #[allow(dead_code)]
    pub(crate) cache_hits: AtomicU64,
    /// Number of cache misses (when cache is enabled).
    #[allow(dead_code)]
    pub(crate) cache_misses: AtomicU64,
    /// Number of tensors pinned in the cache.
    #[allow(dead_code)]
    pub(crate) pinned_tensors: AtomicU64,
    /// Total size of pinned tensors in bytes.
    #[allow(dead_code)]
    pub(crate) pinned_bytes: AtomicU64,
}

impl RuntimeStats {
    /// Create a new zeroed stats counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a tensor access.
    pub(crate) fn record_access(&self, bytes: u64) {
        self.tensor_accesses.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a cache hit.
    pub(crate) fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss.
    pub(crate) fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Total tensor access calls made.
    pub fn tensor_accesses(&self) -> u64 {
        self.tensor_accesses.load(Ordering::Relaxed)
    }

    /// Cumulative bytes read from the mmap.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Cache hit count.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Cache miss count.
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Cache hit ratio (0.0 to 1.0), or `None` if no accesses.
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        let hits = self.cache_hits();
        let misses = self.cache_misses();
        let total = hits + misses;
        if total == 0 {
            None
        } else {
            Some(hits as f64 / total as f64)
        }
    }
}
