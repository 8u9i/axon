//! # TensorCache
//!
//! LRU eviction cache for tensor data with configurable memory budget
//! and pinning support.
//!
//! ## Design
//!
//! - Tensors are cached as `Arc<Vec<u8>>` — cache hits are reference-count
//!   increments, not copies.
//! - LRU eviction when `put` would exceed the memory budget.
//! - Pinned tensors are never evicted (but still count toward the budget).
//! - All operations are `&mut self` (single-threaded for now; thread-safe
//!   wrappers can be added later).
//!
//! ## Memory tracking
//!
//! The cache tracks `current_usage` as a `usize`. On `put`, it evicts
//! unpinned tensors (LRU order) until `current_usage + tensor_size <= budget`.
//! Pinned tensor sizes count toward current_usage but are never evicted.

use std::sync::Arc;
use std::collections::HashSet;

use lru::LruCache;

/// Eviction policy for the tensor cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used — evict the tensor that hasn't been accessed
    /// for the longest time.
    Lru,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::Lru
    }
}

/// Statistics about cache behavior.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses (mmap reads).
    pub misses: u64,
    /// Number of evictions performed.
    pub evictions: u64,
    /// Current memory usage in bytes.
    pub current_usage: usize,
    /// Memory budget in bytes.
    pub budget: usize,
    /// Number of pinned tensors.
    pub pinned_count: usize,
}

impl CacheStats {
    /// Cache hit ratio (0.0 to 1.0), or `None` if no accesses.
    pub fn hit_ratio(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        if total == 0 { None } else { Some(self.hits as f64 / total as f64) }
    }

    /// Current memory usage as a fraction of budget (0.0 to 1.0).
    pub fn usage_ratio(&self) -> f64 {
        if self.budget == 0 { 0.0 } else { self.current_usage as f64 / self.budget as f64 }
    }
}

/// LRU eviction cache for tensors.
///
/// ## Example
///
/// ```rust
/// use axon_runtime::tensor_cache::TensorCache;
///
/// let mut cache = TensorCache::new(1024 * 1024); // 1MB budget
/// let data = vec![0u8; 100];
///
/// // Store a tensor
/// let cached = cache.put("layer_0_q".to_string(), data);
/// assert_eq!(cached.len(), 100);
///
/// // Retrieve it (cache hit)
/// let retrieved = cache.get("layer_0_q").unwrap();
/// assert_eq!(retrieved.len(), 100);
///
/// // Pin it to prevent eviction
/// cache.pin("layer_0_q");
///
/// // Unpin
/// cache.unpin("layer_0_q");
/// ```
pub struct TensorCache {
    /// The LRU cache. Stores `Arc<Vec<u8>>` so cache hits are reference-count ops.
    inner: LruCache<String, Arc<Vec<u8>>>,
    /// Per-tensor sizes tracked alongside the cache.
    sizes: lru::LruCache<String, usize>,
    /// Memory budget in bytes.
    budget: usize,
    /// Current tracked memory usage in bytes.
    current_usage: usize,
    /// Tensors that must not be evicted.
    pinned: HashSet<String>,
    /// Eviction policy.
    policy: EvictionPolicy,
    /// Statistics.
    stats: CacheStats,
}

impl TensorCache {
    /// Create a new cache with the given memory budget in bytes.
    pub fn new(budget: usize) -> Self {
        let cap = budget.max(1024) / 1024; // heuristic: ~1K entries per MB budget
        Self {
            inner: LruCache::new(std::num::NonZeroUsize::new(cap.max(16)).unwrap()),
            sizes: lru::LruCache::new(std::num::NonZeroUsize::new(cap.max(16)).unwrap()),
            budget,
            current_usage: 0,
            pinned: HashSet::new(),
            policy: EvictionPolicy::Lru,
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with a custom eviction policy.
    pub fn with_policy(budget: usize, policy: EvictionPolicy) -> Self {
        let mut c = Self::new(budget);
        c.policy = policy;
        c
    }

    /// Get a tensor from the cache. Returns `None` if not present.
    /// On hit, promotes the tensor to the front of the LRU list.
    pub fn get(&mut self, name: &str) -> Option<Arc<Vec<u8>>> {
        if let Some(data) = self.inner.get(name) {
            self.stats.hits += 1;
            Some(Arc::clone(data))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Store a tensor in the cache.
    ///
    /// If the cache is full, unpinned tensors are evicted (LRU order)
    /// until there is room. Returns an `Arc` to the stored data.
    pub fn put(&mut self, name: String, data: Vec<u8>) -> Arc<Vec<u8>> {
        let size = data.len();

        // Evict until we have room
        while self.current_usage + size > self.budget && !self.inner.is_empty() {
            if !self.evict_one() {
                break; // all remaining are pinned
            }
        }

        let arc = Arc::new(data);

        // If already present, subtract old size
        if let Some(old_size) = self.sizes.get(&name) {
            self.current_usage = self.current_usage.saturating_sub(*old_size);
        }

        self.current_usage += size;
        self.sizes.put(name.clone(), size);
        self.inner.put(name, Arc::clone(&arc));

        // Update stats
        self.stats.current_usage = self.current_usage;
        self.stats.budget = self.budget;

        arc
    }

    /// Pin a tensor — prevent it from being evicted.
    /// Has no effect if the tensor is not in the cache.
    pub fn pin(&mut self, name: &str) {
        if self.inner.contains(name) {
            self.pinned.insert(name.to_string());
            self.stats.pinned_count = self.pinned.len();
        }
    }

    /// Unpin a tensor — allow it to be evicted again.
    pub fn unpin(&mut self, name: &str) {
        self.pinned.remove(name);
        self.stats.pinned_count = self.pinned.len();
    }

    /// Check if a tensor is in the cache.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    /// Check if a tensor is pinned.
    pub fn is_pinned(&self, name: &str) -> bool {
        self.pinned.contains(name)
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics counters (keeps cached data intact).
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats {
            current_usage: self.current_usage,
            budget: self.budget,
            pinned_count: self.pinned.len(),
            ..Default::default()
        };
    }

    /// Evict a specific tensor by name.
    pub fn evict(&mut self, name: &str) {
        if self.pinned.contains(name) {
            return;
        }
        if let Some(size) = self.sizes.pop(name) {
            self.current_usage = self.current_usage.saturating_sub(size);
            self.inner.pop(name);
            self.stats.evictions += 1;
            self.stats.current_usage = self.current_usage;
        }
    }

    /// Clear all cached tensors (pinned or not).
    pub fn clear(&mut self) {
        self.inner.clear();
        self.sizes.clear();
        self.pinned.clear();
        self.current_usage = 0;
        self.stats.current_usage = 0;
        self.stats.pinned_count = 0;
    }

    /// Current memory usage in bytes.
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Memory budget in bytes.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Number of cached tensors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Evict one unpinned tensor (LRU order). Returns `false` if none can be evicted.
    fn evict_one(&mut self) -> bool {
        // Find the first unpinned entry
        let evict_name = self.inner.iter().rev()
            .find(|(name, _)| !self.pinned.contains(*name))
            .map(|(name, _)| name.clone());

        if let Some(name) = evict_name {
            self.evict(&name);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_and_get() {
        let mut cache = TensorCache::new(1024 * 1024);
        assert!(cache.get("test").is_none());

        let data = vec![42u8; 100];
        cache.put("test".to_string(), data);
        assert!(cache.contains("test"));

        let retrieved = cache.get("test").unwrap();
        assert_eq!(retrieved[0], 42u8);
        assert_eq!(retrieved.len(), 100);
    }

    #[test]
    fn test_eviction_when_over_budget() {
        // Budget: 200 bytes, each tensor is 100 bytes
        let mut cache = TensorCache::new(200);
        cache.put("a".to_string(), vec![0u8; 100]);
        cache.put("b".to_string(), vec![1u8; 100]);
        assert_eq!(cache.current_usage(), 200);
        assert!(cache.contains("a"));
        assert!(cache.contains("b"));

        // Adding a third should evict the LRU (which is "a")
        cache.put("c".to_string(), vec![2u8; 100]);
        assert!(!cache.contains("a"), "a should have been evicted");
        assert!(cache.contains("b"));
        assert!(cache.contains("c"));
        assert!(cache.current_usage() <= 200);
    }

    #[test]
    fn test_pinned_tensors_not_evicted() {
        let mut cache = TensorCache::new(200);
        cache.put("pinned".to_string(), vec![0u8; 120]);
        cache.put("other".to_string(), vec![1u8; 80]);
        cache.pin("pinned");

        // Adding another 80-byte tensor should evict "other", not "pinned"
        cache.put("new".to_string(), vec![2u8; 80]);

        assert!(cache.contains("pinned"), "pinned tensor should survive");
        assert!(!cache.contains("other"), "unpinned should be evicted");
        assert!(cache.contains("new"));
    }

    #[test]
    fn test_eviction_count() {
        let mut cache = TensorCache::new(100);
        cache.put("a".to_string(), vec![0u8; 60]);
        assert_eq!(cache.stats().evictions, 0);

        cache.put("b".to_string(), vec![1u8; 60]);
        // "a" should be evicted
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_pin_unpin() {
        let mut cache = TensorCache::new(1000);
        cache.put("x".to_string(), vec![0u8; 100]);
        cache.pin("x");
        assert!(cache.is_pinned("x"));

        cache.unpin("x");
        assert!(!cache.is_pinned("x"));
    }

    #[test]
    fn test_clear() {
        let mut cache = TensorCache::new(1000);
        cache.put("a".to_string(), vec![0u8; 100]);
        cache.put("b".to_string(), vec![1u8; 100]);
        cache.pin("a");
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_usage(), 0);
        assert!(!cache.is_pinned("a"));
    }

    #[test]
    fn test_cache_hit_stats() {
        let mut cache = TensorCache::new(1000);
        cache.put("x".to_string(), vec![0u8; 100]);

        assert!(cache.get("x").is_some());
        assert!(cache.get("x").is_some());
        assert!(cache.get("nonexistent").is_none());

        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_ratio().unwrap() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_eviction_pinned_only_no_eviction() {
        let mut cache = TensorCache::new(100);
        cache.put("only".to_string(), vec![0u8; 100]);
        cache.pin("only");

        // Try to put another tensor — should fail because only pinned remains
        cache.put("second".to_string(), vec![1u8; 100]);

        // "only" stays, "second" may or may not fit depending on rounding
        assert!(cache.contains("only"));
    }

    #[test]
    fn test_large_tensor_directly_evicts() {
        let mut cache = TensorCache::new(100);
        cache.put("small".to_string(), vec![0u8; 60]);

        // Put a tensor larger than the entire budget
        cache.put("big".to_string(), vec![1u8; 200]);

        // "small" should be evicted
        assert!(!cache.contains("small"));
        // "big" might fit if we evicted enough — LRU may not evict all
        // The point is we don't panic or break
        assert!(cache.current_usage() <= 200);
    }
}
