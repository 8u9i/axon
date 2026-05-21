//! Key-Value cache for autoregressive transformer generation.

use crate::ops;

/// Cached key and value tensors for the KV cache.
///
/// Shape: [n_layers, n_kv_heads, max_seq_len, head_dim]
/// We grow the cache as needed during generation.
pub struct KVCache {
    /// Keys per layer: [n_kv_heads * max_seq_len * head_dim]
    keys: Vec<Vec<f32>>,
    /// Values per layer: [n_kv_heads * max_seq_len * head_dim]
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
}

impl KVCache {
    /// Create a new KV cache.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layer_size = n_kv_heads * max_seq_len * head_dim;
        let keys = (0..n_layers).map(|_| vec![0.0f32; layer_size]).collect();
        let values = (0..n_layers).map(|_| vec![0.0f32; layer_size]).collect();

        Self {
            keys,
            values,
            n_layers,
            n_kv_heads,
            head_dim,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Get the current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Check if the cache is full.
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }

    /// Store key and value for a single token position in a specific layer.
    ///
    /// Call this once per layer at each token position.
    pub fn push_layer(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_len = self.n_kv_heads * self.head_dim;
        assert_eq!(key.len(), kv_len);
        assert_eq!(value.len(), kv_len);
        assert!(self.seq_len < self.max_seq_len, "KV cache full");
        let layer_offset = self.seq_len * self.n_kv_heads * self.head_dim;
        ops::copy(&mut self.keys[layer][layer_offset..layer_offset + kv_len], key);
        ops::copy(&mut self.values[layer][layer_offset..layer_offset + kv_len], value);
    }

    /// Advance the sequence length after all layers have been stored for this position.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Get all cached keys for a specific layer.
    /// Returns a slice [n_kv_heads * seq_len * head_dim].
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
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Get the number of layers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Get the number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}
