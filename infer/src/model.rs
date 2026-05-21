//! Model configuration and weight loading.
//!
//! Reads an `.axon` file, detects the architecture from the manifest,
//! and maps the tensor names to model weights.

use std::collections::HashMap;

use axon_core::DType;
use axon_runtime::AxonRuntime;

use crate::dtype;

/// Architecture-specific tensor name mapping.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture name (from manifest).
    pub architecture: String,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Hidden dimension size.
    pub dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub n_kv_heads: usize,
    /// Intermediate dimension for MLP (FFN).
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Max context length.
    pub ctx_len: usize,
    /// Embedding dimension per head.
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// Whether to use RoPE
    pub use_rope: bool,
}

impl ModelConfig {
    /// Parse model configuration from the manifest metadata.
    pub fn from_runtime(rt: &AxonRuntime) -> Result<Self, String> {
        let manifest = rt.manifest();
        let hp = &manifest.hyperparameters;

        let architecture = rt.architecture().to_string();

        // Common hyperparameter names across architectures
        let get_u64 = |keys: &[&str], default: u64| -> u64 {
            for key in keys {
                if let Some(v) = hp.get(*key).and_then(|v| v.as_u64()) {
                    return v;
                }
                // Try as string that can be parsed
                if let Some(v) = hp.get(*key).and_then(|v| v.as_str()) {
                    if let Ok(n) = v.parse::<u64>() {
                        return n;
                    }
                }
            }
            default
        };

        let n_layers = get_u64(&["block_count", "n_layer", "num_hidden_layers", "n_layers"], 0) as usize;
        let dim = get_u64(&["embedding_length", "n_embd", "hidden_size", "d_model", "dim"], 0) as usize;
        let n_heads = get_u64(&["attention.head_count", "n_head", "num_attention_heads", "n_heads"], 0) as usize;
        let n_kv_heads = get_u64(&["attention.head_count_kv", "n_head_kv", "num_key_value_heads", "n_kv_heads"], n_heads as u64) as usize;
        let hidden_dim = get_u64(&["feed_forward_length", "n_ff", "intermediate_size", "hidden_dim"], (dim * 4) as u64) as usize;
        let vocab_size = get_u64(&["vocab_size", "vocabulary_size", "n_vocab"], 32000) as usize;
        let ctx_len = get_u64(&["context_length", "n_ctx", "max_position_embeddings", "seq_len"], 4096) as usize;
        let norm_eps = hp.get("attention.layer_norm_rms_epsilon")
            .or_else(|| hp.get("rms_norm_eps"))
            .or_else(|| hp.get("layer_norm_epsilon"))
            .or_else(|| hp.get("norm_epsilon"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let head_dim = dim / n_heads;

        log::info!(
            "Detected architecture: {} (layers={}, dim={}, heads={}, kv_heads={}, hidden_dim={}, vocab={}, ctx={})",
            architecture, n_layers, dim, n_heads, n_kv_heads, hidden_dim, vocab_size, ctx_len
        );

        Ok(Self {
            architecture,
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            hidden_dim,
            vocab_size,
            ctx_len,
            head_dim,
            norm_eps,
            use_rope: true,
        })
    }

    /// Get the tensor name prefix for layer weights in this architecture.
    #[allow(dead_code)]
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        // Most architectures use "blk.{idx}" or "layers.{idx}" or "model.layers.{idx}"
        // GGUF names from llama.cpp use "blk.{idx}" prefix
        format!("blk.{}", layer_idx)
    }
}

/// A loaded model with all tensors accessible by logical name.
pub struct LoadedModel {
    pub config: ModelConfig,
    pub rt: AxonRuntime,
    /// Column major quantization info: which tensors are quantized and their dtypes.
    tensor_dtypes: HashMap<String, DType>,
}

impl LoadedModel {
    /// Load a model from an .axon file.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let rt = AxonRuntime::open(path)
            .map_err(|e| format!("failed to open model: {e}"))?;
        let config = ModelConfig::from_runtime(&rt)?;

        // Index tensor dtypes
        let mut tensor_dtypes = HashMap::new();
        for info in rt.tensors() {
            tensor_dtypes.insert(info.name.clone(), info.dtype);
        }

        Ok(Self { config, rt, tensor_dtypes })
    }

    /// Get the dtype of a tensor.
    pub fn dtype(&self, name: &str) -> Option<DType> {
        self.tensor_dtypes.get(name).copied()
    }

    /// Get a tensor's raw bytes.
    pub fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        self.rt.tensor(name).map_err(|e| format!("tensor '{}' not found: {e}", name))
    }

    /// Get a tensor's raw f32 view (dequantized if needed).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let raw = self.raw(name)?;
        let dtype = self.dtype(name).unwrap_or(DType::F32);
        Ok(dtype::dequantize_tensor(&raw, dtype))
    }

    /// Get dequantized weights for a specific layer and projection.
    ///
    /// Projection names follow the GGUF naming convention used by llama.cpp:
    /// - `attn_q`, `attn_k`, `attn_v`, `attn_output`
    /// - `ffn_gate`, `ffn_up`, `ffn_down`
    /// - `attn_norm`, `ffn_norm`
    pub fn layer_weight(&self, layer: usize, proj: &str) -> Result<Vec<f32>, String> {
        // Try GGUF naming first: blk.{layer}.{proj}.weight
        let gguf_name = format!("blk.{}.{}.weight", layer, proj);
        if self.tensor_dtypes.contains_key(&gguf_name) {
            return self.f32(&gguf_name);
        }
        // Try HF naming: model.layers.{layer}.self_attn.{proj}_proj.weight
        let hf_name = format!("model.layers.{}.self_attn.{}_proj.weight", layer, match proj {
            "attn_q" => "q",
            "attn_k" => "k",
            "attn_v" => "v",
            "attn_output" => "o",
            _ => return Err(format!("unknown projection '{}' for layer {}", proj, layer)),
        });
        if self.tensor_dtypes.contains_key(&hf_name) {
            return self.f32(&hf_name);
        }
        // Try another common pattern: layers.{idx}.{proj}
        let simple_name = format!("layers.{}.{}", layer, proj);
        if self.tensor_dtypes.contains_key(&simple_name) {
            return self.f32(&simple_name);
        }

        Err(format!(
            "weight not found: layer {layer}, projection '{proj}' (tried: {gguf_name}, {hf_name})"
        ))
    }

    /// Get a standalone weight (not per-layer) like embedding, norm, lm_head.
    pub fn weight(&self, name: &str) -> Result<Vec<f32>, String> {
        // Try common names
        let candidates = [
            name.to_string(),
            format!("{}.weight", name),
            format!("tok_embeddings.weight"),
            format!("output.weight"),
            format!("{}.weight", name.replace("_weight", "")),
        ];

        // Common standalone weights and their GGUF/HF names
        let known_mappings: &[(&str, &[&str])] = &[
            ("tok_embeddings", &["token_embd.weight", "model.embed_tokens.weight", "gpt_neox.embed_in.weight", "wte.weight", "tok_embeddings.weight"]),
            ("norm", &["output_norm.weight", "model.norm.weight", "gpt_neox.final_layer_norm.weight", "ln_f.weight", "norm.weight"]),
            ("output", &["output.weight", "lm_head.weight", "embed_out.weight", "model.embed_tokens.weight"]),
        ];

        for &(logical, variants) in known_mappings {
            if logical == name || format!("{logical}.weight") == name || format!("{logical}_weight") == name {
                for variant in variants {
                    if self.tensor_dtypes.contains_key(*variant) {
                        return self.f32(variant);
                    }
                }
            }
        }

        // Direct lookup
        if self.tensor_dtypes.contains_key(&candidates[0]) {
            return self.f32(&candidates[0]);
        }

        Err(format!(
            "standalone weight '{}' not found (available tensors: {:?})",
            name,
            self.rt.tensor_names()
        ))
    }

    /// Check if a tensor name exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_dtypes.contains_key(name)
    }

    /// List all tensor names for debugging.
    pub fn tensor_names(&self) -> Vec<String> {
        self.rt.tensor_names().into_iter().map(|s| s.to_string()).collect()
    }
}
