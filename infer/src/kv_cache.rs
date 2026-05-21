//! Key-Value cache for autoregressive transformer generation.

use crate::ops;

const GROW_SIZE: usize = 1024;

/// Cached key and value tensors for the KV cache.
///
/// Shape: [n_layers, n_kv_heads, max_seq_len, head_dim]
/// Grows dynamically as tokens are added.
pub struct KVCache {
    /// Keys per layer: grows as tokens are added
    keys: Vec<Vec<f32>>,
    /// Values per layer: grows as tokens are added
    values: Vec<Vec<f32>>,
    /// Number of layers
    n_layers: usize,
    /// Number of KV heads
    n_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Current sequence length (number of tokens cached)
    seq_len: usize,
    /// Maximum capacity
    max_seq_len: usize,
    /// Allocated capacity per layer (in tokens)
    capacity: usize,
}

impl KVCache {
    /// Create a new KV cache with dynamic growth.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, _max_seq_len: usize) -> Self {
        // Start small — grow as needed
        let initial_cap = GROW_SIZE.min(4096);
        let layer_size = n_kv_heads * initial_cap * head_dim;
        let keys = (0..n_layers).map(|_| vec![0.0f32; layer_size]).collect();
        let values = (0..n_layers).map(|_| vec![0.0f32; layer_size]).collect();

        Self {
            keys, values,
            n_layers, n_kv_heads, head_dim,
            seq_len: 0, max_seq_len: _max_seq_len,
            capacity: initial_cap,
        }
    }

    /// Ensure capacity for the given number of tokens.
    fn ensure_capacity(&mut self, needed: usize) {
        if needed <= self.capacity { return; }
        let new_cap = (needed + GROW_SIZE - 1) / GROW_SIZE * GROW_SIZE;
        let new_cap = new_cap.min(self.max_seq_len);
        let new_layer_size = self.n_kv_heads * new_cap * self.head_dim;
        for l in 0..self.n_layers {
            let old_size = self.keys[l].len();
            if old_size < new_layer_size {
                self.keys[l].resize(new_layer_size, 0.0f32);
                self.values[l].resize(new_layer_size, 0.0f32);
            }
        }
        self.capacity = new_cap;
    }

    /// Get the current sequence length.
    pub fn seq_len(&self) -> usize { self.seq_len }

    /// Check if the cache is full.
    pub fn is_full(&self) -> bool { self.seq_len >= self.max_seq_len }

    /// Store key and value for a single token position in a specific layer.
    pub fn push_layer(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_len = self.n_kv_heads * self.head_dim;
        assert_eq!(key.len(), kv_len);
        assert_eq!(value.len(), kv_len);
        assert!(self.seq_len < self.max_seq_len, "KV cache full");
        self.ensure_capacity(self.seq_len + 1);
        let layer_offset = self.seq_len * self.n_kv_heads * self.head_dim;
        ops::copy(&mut self.keys[layer][layer_offset..layer_offset + kv_len], key);
        ops::copy(&mut self.values[layer][layer_offset..layer_offset + kv_len], value);
    }

    /// Advance the sequence length after all layers have been stored for this position.
    pub fn advance(&mut self) { self.seq_len += 1; }

    /// Get all cached keys for a specific layer.
    pub fn get_keys(&self, layer: usize) -> &[f32] {
        let len = self.n_kv_heads * self.seq_len * self.head_dim;
        &self.keys[layer][..len]
    }

    /// Get all cached values for a specific layer.
    pub fn get_values(&self, layer: usize) -> &[f32] {
        let len = self.n_kv_heads * self.seq_len * self.head_dim;
        &self.values[layer][..len]
    }

    /// Get the key for a single head and position.
    pub fn get_key_at(&self, layer: usize, head: usize, pos: usize) -> &[f32] {
        let offset = (pos * self.n_kv_heads + head) * self.head_dim;
        &self.keys[layer][offset..offset + self.head_dim]
    }

    /// Get the value for a single head and position.
    pub fn get_value_at(&self, layer: usize, head: usize, pos: usize) -> &[f32] {
        let offset = (pos * self.n_kv_heads + head) * self.head_dim;
        &self.values[layer][offset..offset + self.head_dim]
    }

    /// Reset the cache.
    pub fn clear(&mut self) { self.seq_len = 0; }

    pub fn n_layers(&self) -> usize { self.n_layers }
    pub fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
}
