//! Transformer forward pass.
//!
//! Implements the full forward pass for decoder-only transformer models
//! (Llama-family: Llama, Mistral, Qwen, Gemma, etc.).

use axon_core::DType;

use crate::dtype;
use crate::kv_cache::KVCache;
use crate::model::LoadedModel;
use crate::ops;
use crate::quantized;

/// A single transformer layer's weights, loaded from the model.
pub struct TransformerLayerWeights {
    /// Attention weights
    pub wq: Vec<f32>,  // [dim, n_heads * head_dim]
    pub wk: Vec<f32>,  // [dim, n_kv_heads * head_dim]
    pub wv: Vec<f32>,  // [dim, n_kv_heads * head_dim]
    pub wo: Vec<f32>,  // [n_heads * head_dim, dim]
    /// MLP weights
    pub w1: Vec<f32>,  // gate: [dim, hidden_dim]
    pub w2: Vec<f32>,  // down: [hidden_dim, dim]
    pub w3: Vec<f32>,  // up: [dim, hidden_dim]
    /// RMS norm weights
    pub attention_norm: Vec<f32>,  // [dim]
    pub ffn_norm: Vec<f32>,        // [dim]
    /// Raw bytes for quantized matmul (used instead of dequantized f32 if present)
    pub raw_wq: Option<Vec<u8>>,
    pub raw_wk: Option<Vec<u8>>,
    pub raw_wv: Option<Vec<u8>>,
    pub raw_wo: Option<Vec<u8>>,
    pub raw_w1: Option<Vec<u8>>,
    pub raw_w2: Option<Vec<u8>>,
    pub raw_w3: Option<Vec<u8>>,
    /// DType for raw weights
    pub dtype: DType,
}

/// Global model weights.
pub struct ModelWeights {
    pub config: crate::model::ModelConfig,
    /// Token embedding table: [vocab_size, dim]
    pub tok_embeddings: Vec<f32>,
    pub raw_tok_embeddings: Option<Vec<u8>>,
    /// Per-layer weights
    pub layers: Vec<TransformerLayerWeights>,
    /// Final RMS norm: [dim]
    pub norm: Vec<f32>,
    /// Output projection (lm_head): [vocab_size, dim] or shared with embeddings
    pub output: Option<Vec<f32>>,
    pub raw_output: Option<Vec<u8>>,
    /// Whether output weight is shared with embeddings (tied)
    pub output_shared: bool,
    /// DType for embeddings and output
    pub embed_dtype: DType,
    pub output_dtype: DType,
}

impl ModelWeights {
    /// Load all weights from a loaded model.
    pub fn load(model: &LoadedModel) -> Result<Self, String> {
        let config = &model.config;

        // Load token embeddings
        let (tok_emb, raw_tok_emb, embed_dtype) = Self::load_weight_flexible(
            model, "tok_embeddings",
            &["token_embd.weight", "model.embed_tokens.weight", "gpt_neox.embed_in.weight", "wte.weight"],
            config.vocab_size * config.dim,
        )?;
        let tok_embeddings = if tok_emb.len() == config.vocab_size * config.dim {
            tok_emb
        } else {
            return Err(format!(
                "token embeddings size mismatch: expected {} elements, got {}",
                config.vocab_size * config.dim,
                tok_emb.len()
            ));
        };

        // Load output projection
        let (output, raw_output, output_dtype) = Self::load_weight_flexible(
            model, "output",
            &["output.weight", "lm_head.weight", "embed_out.weight"],
            config.vocab_size * config.dim,
        )?;
        let output_shared = model.has_tensor("output.weight") && output.is_empty()
            || model.has_tensor("lm_head.weight") && output.is_empty()
            || !model.has_tensor("output.weight") && !model.has_tensor("lm_head.weight");

        // Load final norm
        let norm = Self::load_norm(model, "norm")?;

        // Load each layer
        let mut layers = Vec::with_capacity(config.n_layers);
        for layer_idx in 0..config.n_layers {
            let layer = Self::load_layer(model, layer_idx)?;
            layers.push(layer);
        }

        Ok(Self {
            config: config.clone(),
            tok_embeddings,
            raw_tok_embeddings: raw_tok_emb,
            layers,
            norm,
            output: if output.is_empty() { None } else { Some(output) },
            raw_output,
            output_shared,
            embed_dtype,
            output_dtype,
        })
    }

    /// Flexible weight loading with multiple name candidates.
    fn load_weight_flexible(
        model: &LoadedModel,
        _logical: &str,
        candidates: &[&str],
        _expected_elements: usize,
    ) -> Result<(Vec<f32>, Option<Vec<u8>>, DType), String> {
        for &name in candidates {
            if model.has_tensor(name) {
                let dtype = model.dtype(name).unwrap_or(DType::F32);
                let raw = model.raw(name)?;
                let f32_data = dtype::dequantize_tensor(&raw, dtype);
                return Ok((f32_data, Some(raw), dtype));
            }
        }
        Ok((Vec::new(), None, DType::F32))
    }

    fn load_norm(model: &LoadedModel, _logical: &str) -> Result<Vec<f32>, String> {
        // Try various norm names
        let norm_candidates = [
            "output_norm.weight",
            "model.norm.weight",
            "gpt_neox.final_layer_norm.weight",
            "ln_f.weight",
            "norm.weight",
            "token_embd_norm.weight",
        ];
        for &name in &norm_candidates {
            if model.has_tensor(name) {
                return model.f32(name);
            }
        }
        // If no norm found, return ones
        log::warn!("No output norm found, using identity");
        Ok(vec![1.0; model.config.dim])
    }

    fn load_layer(model: &LoadedModel, layer_idx: usize) -> Result<TransformerLayerWeights, String> {
        let config = &model.config;
        let prefix = format!("blk.{}", layer_idx);

        // GGUF names for this layer
        let wq_names = [format!("{}.attn_q.weight", prefix), format!("{prefix}.self_attn.q_proj.weight"), format!("model.layers.{layer_idx}.self_attn.q_proj.weight")];
        let wk_names = [format!("{}.attn_k.weight", prefix), format!("{prefix}.self_attn.k_proj.weight")];
        let wv_names = [format!("{}.attn_v.weight", prefix), format!("{prefix}.self_attn.v_proj.weight")];
        let wo_names = [format!("{}.attn_output.weight", prefix), format!("{prefix}.self_attn.o_proj.weight")];
        let w1_names = [format!("{}.ffn_gate.weight", prefix), format!("{prefix}.mlp.gate_proj.weight")];
        let w2_names = [format!("{}.ffn_down.weight", prefix), format!("{prefix}.mlp.down_proj.weight")];
        let w3_names = [format!("{}.ffn_up.weight", prefix), format!("{prefix}.mlp.up_proj.weight")];
        let attn_norm_names = [format!("{}.attn_norm.weight", prefix), format!("{prefix}.input_layernorm.weight")];
        let ffn_norm_names = [format!("{}.ffn_norm.weight", prefix), format!("{prefix}.post_attention_layernorm.weight")];

        let name_for = |names: &[String]| -> String { names.iter().find(|n| model.has_tensor(n)).cloned().unwrap_or_default() };

        let wq_name = name_for(&wq_names);
        let wk_name = name_for(&wk_names);
        let wv_name = name_for(&wv_names);
        let wo_name = name_for(&wo_names);
        let w1_name = name_for(&w1_names);
        let w2_name = name_for(&w2_names);
        let w3_name = name_for(&w3_names);
        let attn_norm_name = name_for(&attn_norm_names);
        let ffn_norm_name = name_for(&ffn_norm_names);

        let load_weight = |name: &str| -> (Vec<f32>, Option<Vec<u8>>, DType) {
            if name.is_empty() {
                return (Vec::new(), None, DType::F32);
            }
            let dtype = model.dtype(name).unwrap_or(DType::F32);
            let raw = model.raw(name).unwrap_or_default();
            let f32_data = dtype::dequantize_tensor(&raw, dtype);
            (f32_data, Some(raw), dtype)
        };

        let load_norm = |name: &str| -> Vec<f32> {
            if name.is_empty() {
                return vec![1.0; config.dim];
            }
            model.f32(name).unwrap_or_else(|_| vec![1.0; config.dim])
        };

        let (wq, raw_wq, dtype) = load_weight(&wq_name);
        let (wk, raw_wk, _) = load_weight(&wk_name);
        let (wv, raw_wv, _) = load_weight(&wv_name);
        let (wo, raw_wo, _) = load_weight(&wo_name);
        let (w1, raw_w1, _) = load_weight(&w1_name);
        let (w2, raw_w2, _) = load_weight(&w2_name);
        let (w3, raw_w3, _) = load_weight(&w3_name);
        let attention_norm = load_norm(&attn_norm_name);
        let ffn_norm = load_norm(&ffn_norm_name);

        Ok(TransformerLayerWeights {
            wq, wk, wv, wo, w1, w2, w3,
            attention_norm, ffn_norm,
            raw_wq, raw_wk, raw_wv, raw_wo,
            raw_w1, raw_w2, raw_w3,
            dtype,
        })
    }
}

/// Run the forward pass for a single token through all layers.
///
/// Input: x [dim] — the token embedding (or previous layer's output)
/// Output: logits [vocab_size]
pub fn forward(
    x: &mut [f32],          // [dim] — will be modified in-place through layers
    logits: &mut [f32],     // [vocab_size] — output
    weights: &ModelWeights,
    cache: &mut KVCache,
    pos: usize,             // Token position in the sequence
) {
    let config = &weights.config;
    let n_q_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let dim = config.dim;

    // Ensure input size
    assert_eq!(x.len(), dim, "Input size mismatch");

    // ── Embedding lookup is done before calling forward ──
    // x should already be the embedding for the current token

    // ── Process each layer ──
    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // --- Attention ---
        // RMSNorm on x
        let mut attn_input = x.to_vec();
        ops::rms_norm(&mut attn_input, &layer.attention_norm, config.norm_eps);

        // QKV projections: Q = x @ Wq, K = x @ Wk, V = x @ Wv
        let q_len = n_q_heads * head_dim;
        let kv_len = n_kv_heads * head_dim;
        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];

        // Use quantized matvec if available, else f32 matvec
        if let Some(ref raw) = layer.raw_wq {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            quantized::quantized_matvec(raw, q_len, dim, block_bytes, vals_per_block, &attn_input, &mut q);
        } else {
            ops::matvec(q_len, dim, &layer.wq, &attn_input, &mut q);
        }

        if let Some(ref raw) = layer.raw_wk {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            quantized::quantized_matvec(raw, kv_len, dim, block_bytes, vals_per_block, &attn_input, &mut k);
        } else {
            ops::matvec(kv_len, dim, &layer.wk, &attn_input, &mut k);
        }

        if let Some(ref raw) = layer.raw_wv {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            quantized::quantized_matvec(raw, kv_len, dim, block_bytes, vals_per_block, &attn_input, &mut v);
        } else {
            ops::matvec(kv_len, dim, &layer.wv, &attn_input, &mut v);
        }

        // Apply RoPE to Q and K
        ops::apply_rope_multi(&mut q, pos, n_q_heads, head_dim, 10000.0);
        ops::apply_rope_multi(&mut k, pos, n_kv_heads, head_dim, 10000.0);

        // Store K, V in cache
        cache.push_layer(layer_idx, &k, &v);
        // scores[h][qi] = sum over head_dim of q[h][d] * k_cache[h][pos][d]
        // For single token generation (qi=0), just compute attention over all cached positions.
        let cached_len = cache.seq_len();
        let mut attn_scores = vec![0.0f32; n_q_heads * cached_len];

        for h in 0..n_q_heads {
            let q_head = h * head_dim;
            // GQA: map query head to KV head
            let kv_head = h * n_kv_heads / n_q_heads;
            let kv_head = kv_head.min(n_kv_heads - 1);

            for pos_k in 0..cached_len {
                let k_offset = (pos_k * n_kv_heads + kv_head) * head_dim;
                let cache_keys = cache.get_keys(layer_idx);
                let k_slice = &cache_keys[k_offset..k_offset + head_dim];
                let score = ops::dot(&q[q_head..q_head + head_dim], k_slice);
                attn_scores[h * cached_len + pos_k] = score;
            }
        }

        // Scale
        let scale = 1.0 / (head_dim as f32).sqrt();
        for s in attn_scores.iter_mut() {
            *s *= scale;
        }

        // Softmax
        for h in 0..n_q_heads {
            let start = h * cached_len;
            ops::softmax(&mut attn_scores[start..start + cached_len]);
        }

        // --- Attention output ---
        // output[h][d] = sum over positions of scores[h][pos] * v_cache[h][pos][d]
        let mut attn_output = vec![0.0f32; n_q_heads * head_dim];

        for h in 0..n_q_heads {
            let kv_head = h * n_kv_heads / n_q_heads;
            let kv_head = kv_head.min(n_kv_heads - 1);
            let out_start = h * head_dim;

            for pos_k in 0..cached_len {
                let score = attn_scores[h * cached_len + pos_k];
                let v_offset = (pos_k * n_kv_heads + kv_head) * head_dim;
                let cache_values = cache.get_values(layer_idx);
                for d in 0..head_dim {
                    attn_output[out_start + d] += score * cache_values[v_offset + d];
                }
            }
        }

        // Output projection: attn_output @ Wo
        let mut attn_proj = vec![0.0f32; dim];
        if let Some(ref raw) = layer.raw_wo {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            // Wo is [n_heads * head_dim, dim] — need to handle as matvec where rows=dim, cols=q_len
            quantized::quantized_matvec(raw, dim, q_len, block_bytes, vals_per_block, &attn_output, &mut attn_proj);
        } else {
            ops::matvec_transpose(dim, q_len, &layer.wo, &attn_output, &mut attn_proj);
        }

        // Residual connection: x += attn_proj
        ops::add_inplace(x, &attn_proj);

        // --- MLP ---
        let mut mlp_input = x.to_vec();
        ops::rms_norm(&mut mlp_input, &layer.ffn_norm, config.norm_eps);

        // MLP: silu(x @ W1) * (x @ W3) @ W2
        let hidden = config.hidden_dim;
        let mut gate = vec![0.0f32; hidden];
        let mut up = vec![0.0f32; hidden];
        let mut down = vec![0.0f32; dim];

        // gate projection (W1)
        if let Some(ref raw) = layer.raw_w1 {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            quantized::quantized_matvec(raw, hidden, dim, block_bytes, vals_per_block, &mlp_input, &mut gate);
        } else {
            ops::matvec(hidden, dim, &layer.w1, &mlp_input, &mut gate);
        }
        ops::silu_inplace(&mut gate);

        // up projection (W3)
        if let Some(ref raw) = layer.raw_w3 {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            quantized::quantized_matvec(raw, hidden, dim, block_bytes, vals_per_block, &mlp_input, &mut up);
        } else {
            ops::matvec(hidden, dim, &layer.w3, &mlp_input, &mut up);
        }

        // Element-wise multiply: gate * up
        for i in 0..hidden {
            gate[i] *= up[i];
        }

        // down projection (W2): result @ W2
        if let Some(ref raw) = layer.raw_w2 {
            let block_bytes = dtype::block_size_bytes(layer.dtype);
            let vals_per_block = dtype::block_size_values(layer.dtype);
            quantized::quantized_matvec(raw, dim, hidden, block_bytes, vals_per_block, &gate, &mut down);
        } else {
            ops::matvec_transpose(dim, hidden, &layer.w2, &gate, &mut down);
        }

        // Residual
        ops::add_inplace(x, &down);
    }

    // ── Advance KV cache to next position ──
    cache.advance();

    // ── Final RMSNorm ──
    ops::rms_norm(x, &weights.norm, config.norm_eps);

    // ── Output projection (lm_head) ──
    if weights.output_shared {
        // Use token embeddings as output projection
        // logits = x @ emb^T
        let vocab = config.vocab_size;
        ops::matvec_transpose(vocab, dim, &weights.tok_embeddings, x, logits);
    } else if let Some(ref out) = weights.output {
        ops::matvec(config.vocab_size, dim, out, x, logits);
    } else if let Some(ref raw) = weights.raw_output {
        let block_bytes = dtype::block_size_bytes(weights.output_dtype);
        let vals_per_block = dtype::block_size_values(weights.output_dtype);
        quantized::quantized_matvec(raw, config.vocab_size, dim, block_bytes, vals_per_block, x, logits);
    } else {
        // Fallback: shared
        ops::matvec_transpose(config.vocab_size, dim, &weights.tok_embeddings, x, logits);
    }
}
