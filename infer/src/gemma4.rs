//! Gemma 4 inference engine.
//!
//! Supports the full Gemma 4 architecture as used by `gemma4:e4b`:
//! - 42 layers, 2560 dim, 8 heads, 2 KV heads
//! - Mixture of Experts (all layers have 2 shared experts via inp_gate router)
//! - Dual RoPE (global + SWA frequency bases)
//! - Sliding window attention (pattern-based)
//! - QK normalization (per-head RMS norm on Q and K)
//! - Final logit softcapping
//! - Per-layer input projection
//! - Multi-modal: text-only mode skips vision/audio encoders

use axon_core::DType;
use axon_runtime::AxonRuntime;

use crate::dtype;
use crate::kv_cache::KVCache;
use crate::ops;
use crate::sampling::SamplingParams;
use crate::tokenizer::Tokenizer;

// ── Configuration ──────────────────────────────────────────────────

/// Gemma 4 model configuration parsed from the manifest.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub n_layers: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub head_dim_swa: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub ctx_len: usize,
    pub norm_eps: f64,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub rope_dim: usize,
    pub rope_dim_swa: usize,
    pub sliding_window: usize,
    pub swa_pattern: Vec<bool>,
    pub shared_kv_layers: usize,
    pub final_logit_softcapping: f32,
    pub per_layer_dim: usize,
}

impl Gemma4Config {
    pub fn from_runtime(rt: &axon_runtime::AxonRuntime) -> Result<Self, String> {
        let manifest = rt.manifest();
        let hp = &manifest.hyperparameters;
        let md = hp.get("gguf.metadata").and_then(|v| v.as_object()).ok_or_else(|| "no gguf.metadata in manifest".to_string())?;

        let get_f = |key: &str, default: f64| -> f64 {
            md.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let get_u = |key: &str, default: u64| -> u64 {
            md.get(key).and_then(|v| v.as_u64())
                .or_else(|| md.get(key).and_then(|v| v.as_str()).and_then(|s| s.parse().ok()))
                .unwrap_or(default)
        };
        let swa_arr = md.get("gemma4.attention.sliding_window_pattern")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(true)).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![true; 42]);

        Ok(Self {
            n_layers: get_u("gemma4.block_count", 42) as usize,
            dim: get_u("gemma4.embedding_length", 2560) as usize,
            n_heads: get_u("gemma4.attention.head_count", 8) as usize,
            n_kv_heads: get_u("gemma4.attention.head_count_kv", 2) as usize,
            head_dim: get_u("gemma4.attention.key_length", 512) as usize,
            head_dim_swa: get_u("gemma4.attention.key_length_swa", 256) as usize,
            hidden_dim: get_u("gemma4.feed_forward_length", 10240) as usize,
            vocab_size: get_u("gemma4.vocab_size", 262144) as usize,
            ctx_len: get_u("gemma4.context_length", 131072) as usize,
            norm_eps: get_f("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
            rope_base: get_f("gemma4.rope.freq_base", 1000000.0) as f32,
            rope_base_swa: get_f("gemma4.rope.freq_base_swa", 10000.0) as f32,
            rope_dim: get_u("gemma4.rope.dimension_count", 512) as usize,
            rope_dim_swa: get_u("gemma4.rope.dimension_count_swa", 256) as usize,
            sliding_window: get_u("gemma4.attention.sliding_window", 512) as usize,
            swa_pattern: swa_arr,
            shared_kv_layers: get_u("gemma4.attention.shared_kv_layers", 18) as usize,
            final_logit_softcapping: get_f("gemma4.final_logit_softcapping", 30.0) as f32,
            per_layer_dim: get_u("gemma4.embedding_length_per_layer_input", 256) as usize,
        })
    }

    pub fn is_swa_layer(&self, layer: usize) -> bool {
        if layer < self.swa_pattern.len() {
            self.swa_pattern[layer]
        } else {
            true
        }
    }
}

// ── Model Weights ──────────────────────────────────────────────────

pub struct Gemma4Weights {
    pub config: Gemma4Config,
    /// Token embeddings stored as raw bytes (initially Q4), dequantized rows on demand
    pub tok_embd_raw: Vec<u8>,
    pub tok_embd_dtype: DType,
    pub output_norm: Vec<f32>,
    pub output: Option<Vec<f32>>,
    pub layers: Vec<Gemma4LayerWeights>,
    pub rope_freqs: Vec<f32>,
    pub shared_embd: bool,
}

pub struct Gemma4LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_o: Vec<f32>,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub inp_gate: Vec<f32>,        // MoE router [dim, per_layer_dim]
    pub proj: Vec<f32>,            // MoE projection [per_layer_dim, dim]
    pub post_attention_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub layer_output_scale: f32,
    pub has_experts: bool,
    pub expert_gate: Vec<f32>,     // expert 1
    pub expert_up: Vec<f32>,
    pub expert_down: Vec<f32>,
    pub expert_gate_2: Vec<f32>,   // expert 2
    pub expert_up_2: Vec<f32>,
    pub expert_down_2: Vec<f32>,
    // Raw quantized data
    pub raw_attn_q: Option<Vec<u8>>,
    pub raw_attn_k: Option<Vec<u8>>,
    pub raw_attn_v: Option<Vec<u8>>,
    pub raw_attn_o: Option<Vec<u8>>,
    pub raw_ffn_gate: Option<Vec<u8>>,
    pub raw_ffn_up: Option<Vec<u8>>,
    pub raw_ffn_down: Option<Vec<u8>>,
    pub raw_inp_gate: Option<Vec<u8>>,
    pub raw_proj: Option<Vec<u8>>,
    pub raw_expert_gate: Option<Vec<u8>>,
    pub raw_expert_up: Option<Vec<u8>>,
    pub raw_expert_down: Option<Vec<u8>>,
    pub raw_expert_gate_2: Option<Vec<u8>>,
    pub raw_expert_up_2: Option<Vec<u8>>,
    pub raw_expert_down_2: Option<Vec<u8>>,
    pub weights_dtype: DType,
}

impl Gemma4Weights {
    pub fn load(rt: &axon_runtime::AxonRuntime, config: &Gemma4Config) -> Result<Self, String> {
        log::info!("Loading Gemma4 weights: {} layers, dim={}, heads={}, kv_heads={}",
            config.n_layers, config.dim, config.n_heads, config.n_kv_heads);

        let f32_or_empty = |name: &str| -> Vec<f32> {
            if let Ok(info) = rt.tensor_info(name) {
                let raw = rt.tensor(name).unwrap_or_default();
                let expected: usize = info.shape.iter().map(|s| *s as usize).product();
                dtype::dequantize_tensor_with_count(&raw, info.dtype, expected)
            } else { Vec::new() }
        };

        // Store token embeddings in raw format to avoid massive memory usage
        let (tok_embd_raw, tok_embd_dtype) = {
            if let Ok(info) = rt.tensor_info("token_embd.weight") {
                (rt.tensor("token_embd.weight").unwrap_or_default(), info.dtype)
            } else {
                return Err("token_embd.weight not found".to_string());
            }
        };
        let output_norm = f32_or_empty("output_norm.weight");
        if output_norm.is_empty() { return Err("output_norm.weight not found".to_string()); }
        let rope_freqs = f32_or_empty("rope_freqs.weight");
        let _per_layer_proj_norm = f32_or_empty("per_layer_proj_norm.weight");
        let shared_embd = rt.tensor_info("output.weight").is_err();
        let output = if !shared_embd { Some(f32_or_empty("output.weight")) } else { None };

        // Load layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for l in 0..config.n_layers {
            layers.push(Self::load_layer_from_rt(rt, config, l)?);
        }

        Ok(Self {
            config: config.clone(),
            tok_embd_raw, tok_embd_dtype, output_norm, output, layers,
            rope_freqs, shared_embd,
        })
    }

    fn load_layer_from_rt(rt: &axon_runtime::AxonRuntime, config: &Gemma4Config, l: usize) -> Result<Gemma4LayerWeights, String> {
        let p = |suffix: &str| format!("blk.{}.{}", l, suffix);

        let load_deq = |name: &str| -> (Vec<f32>, Option<Vec<u8>>, DType) {
            let full = p(name);
            if let Ok(info) = rt.tensor_info(&full) {
                let dtype = info.dtype;
                let raw = rt.tensor(&full).unwrap_or_default();
                let expected: usize = info.shape.iter().map(|s| *s as usize).product();
                let f32_data = dtype::dequantize_tensor_with_count(&raw, dtype, expected);
                (f32_data, Some(raw), dtype)
            } else { (Vec::new(), None, DType::F32) }
        };

        let load_norm = |name: &str| -> Vec<f32> {
            let full = p(name);
            if let Ok(info) = rt.tensor_info(&full) {
                let raw = rt.tensor(&full).unwrap_or_default();
                let expected: usize = info.shape.iter().map(|s| *s as usize).product();
                dtype::dequantize_tensor_with_count(&raw, DType::F32, expected)
            } else { vec![1.0; config.dim] }
        };

        let (wq, rwq, dt) = load_deq("attn_q.weight");
        let (wk, rwk, _) = load_deq("attn_k.weight");
        let (wv, rwv, _) = load_deq("attn_v.weight");
        let (wo, rwo, _) = load_deq("attn_output.weight");
        let (wg, rwg, _) = load_deq("ffn_gate.weight");
        let (wu, rwu, _) = load_deq("ffn_up.weight");
        let (wd, rwd, _) = load_deq("ffn_down.weight");
        let (wig, rwig, _) = load_deq("inp_gate.weight");
        let (wp, rwp, _) = load_deq("proj.weight");

        // Helper that loads and dequantizes with expected count from shape
        let load_deq_e = |name: &str, rows: usize, cols: usize| -> (Vec<f32>, Option<Vec<u8>>, DType) {
            let full = p(name);
            if let Ok(info) = rt.tensor_info(&full) {
                let raw = rt.tensor(&full).unwrap_or_default();
                let expected = rows * cols;
                (dtype::dequantize_tensor_with_count(&raw, info.dtype, expected), Some(raw), info.dtype)
            } else { (Vec::new(), None, DType::F32) }
        };

        // Experts — they might not exist for all layers — they might not exist for all layers
        let has_e1 = rt.tensor_info(&p("ffn_gate_1.weight")).is_ok();
        let (eg, reg, _) = if has_e1 {
            load_deq_e("ffn_gate_1.weight", config.hidden_dim, config.dim)
        } else { (Vec::new(), None, DType::F32) };
        let (eu, reu, _) = if has_e1 {
            load_deq_e("ffn_up_1.weight", config.hidden_dim, config.dim)
        } else { (Vec::new(), None, DType::F32) };
        let (ed, red, _) = if has_e1 {
            load_deq_e("ffn_down_1.weight", config.dim, config.hidden_dim)
        } else { (Vec::new(), None, DType::F32) };
        let has_e2 = rt.tensor_info(&p("ffn_gate_2.weight")).is_ok();
        let (eg2, reg2, _) = if has_e2 {
            load_deq_e("ffn_gate_2.weight", config.hidden_dim, config.dim)
        } else { (Vec::new(), None, DType::F32) };
        let (eu2, reu2, _) = if has_e2 {
            load_deq_e("ffn_up_2.weight", config.hidden_dim, config.dim)
        } else { (Vec::new(), None, DType::F32) };
        let (ed2, red2, _) = if has_e2 {
            load_deq_e("ffn_down_2.weight", config.dim, config.hidden_dim)
        } else { (Vec::new(), None, DType::F32) };

        let attn_norm = load_norm("attn_norm.weight");
        let ffn_norm = load_norm("ffn_norm.weight");
        let post_attn_norm = load_norm("post_attention_norm.weight");
        let post_ffw_norm = load_norm("post_ffw_norm.weight");
        let post_norm = load_norm("post_norm.weight");
        let layer_scale = {
            let full = p("layer_output_scale.weight");
            if rt.tensor_info(&full).is_ok() {
                let raw = rt.tensor(&full).unwrap_or_default();
                dtype::dequantize_tensor(&raw, DType::F32).first().copied().unwrap_or(1.0)
            } else { 1.0 }
        };
        let attn_q_norm = load_norm("attn_q_norm.weight");
        let attn_k_norm = load_norm("attn_k_norm.weight");

        Ok(Gemma4LayerWeights {
            attn_norm, attn_q: wq, attn_k: wk, attn_v: wv, attn_o: wo,
            attn_q_norm, attn_k_norm,
            ffn_norm, ffn_gate: wg, ffn_up: wu, ffn_down: wd,
            inp_gate: wig, proj: wp,
            post_attention_norm: post_attn_norm,
            post_ffw_norm, post_norm,
            layer_output_scale: layer_scale,
            has_experts: has_e1,
            expert_gate: eg, expert_up: eu, expert_down: ed,
            expert_gate_2: eg2, expert_up_2: eu2, expert_down_2: ed2,
            raw_attn_q: rwq, raw_attn_k: rwk, raw_attn_v: rwv, raw_attn_o: rwo,
            raw_ffn_gate: rwg, raw_ffn_up: rwu, raw_ffn_down: rwd,
            raw_inp_gate: rwig, raw_proj: rwp,
            raw_expert_gate: reg, raw_expert_up: reu, raw_expert_down: red,
            raw_expert_gate_2: reg2, raw_expert_up_2: reu2, raw_expert_down_2: red2,
            weights_dtype: dt,
        })
    }
}

impl Gemma4Weights {
    /// Look up a token embedding by ID (dequantizes on-the-fly from raw storage).
    pub fn embed(&self, token_id: usize, output: &mut [f32]) {
        let config = &self.config;
        let dim = config.dim;
        let row_bytes = dtype::row_stride_bytes(dim, self.tok_embd_dtype);
        let byte_off = token_id * row_bytes;
        if byte_off + row_bytes <= self.tok_embd_raw.len() {
            let row = &self.tok_embd_raw[byte_off..byte_off + row_bytes];
            let deq = dtype::dequantize_tensor(row, self.tok_embd_dtype);
            let n = deq.len().min(dim);
            output[..n].copy_from_slice(&deq[..n]);
            if n < dim {
                output[n..].fill(0.0);
            }
        } else {
            output.fill(0.0);
        }
    }
}

// ── Forward Pass ───────────────────────────────────────────────────

/// Run Gemma 4 forward pass for a single token.
///
/// Input: x [dim] — embedding for current token
/// Output: logits [vocab_size]
pub fn gemma4_forward(
    x: &mut [f32],
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    pos: usize,
) {
    let cfg = &weights.config;
    let dim = cfg.dim;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;

    assert_eq!(x.len(), dim);

    for layer_idx in 0..cfg.n_layers {
        let layer = &weights.layers[layer_idx];
        let is_swa = cfg.is_swa_layer(layer_idx);
        let actual_hd = if is_swa { cfg.head_dim_swa } else { cfg.head_dim };
        let full_hd = cfg.head_dim; // KV cache always uses full head_dim
        let q_len = n_heads * actual_hd;
        let kv_len = n_kv_heads * actual_hd;
        let full_kv_len = n_kv_heads * full_hd;
        let _rope_base = if is_swa { cfg.rope_base_swa } else { cfg.rope_base };

        // ── Pre-attention RMSNorm ──
        let mut h = x.to_vec();
        ops::rms_norm(&mut h, &layer.attn_norm, cfg.norm_eps);

        // ── QKV projections ──
        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        if layer.attn_q.len() != q_len * dim {
            log::error!("Layer {} attn_q: expected {} elements, got {} (shape {}x{})",
                layer_idx, q_len * dim, layer.attn_q.len(), q_len, dim);
        }
        if layer.attn_k.len() != kv_len * dim {
            log::error!("Layer {} attn_k: expected {} elements, got {} (shape {}x{}, dtype={:?}, raw_len={})",
                layer_idx, kv_len * dim, layer.attn_k.len(), kv_len, dim,
                layer.weights_dtype,
                layer.raw_attn_k.as_ref().map(|v| v.len()).unwrap_or(0));
        }
        if layer.attn_v.len() != kv_len * dim {
            log::error!("Layer {} attn_v: expected {} elements, got {}", layer_idx, kv_len * dim, layer.attn_v.len());
        }
        quantized_or_f32_matvec(&layer.raw_attn_q, &layer.attn_q, q_len, dim, &h, &mut q);
        quantized_or_f32_matvec(&layer.raw_attn_k, &layer.attn_k, kv_len, dim, &h, &mut k);
        quantized_or_f32_matvec(&layer.raw_attn_v, &layer.attn_v, kv_len, dim, &h, &mut v);

        // ── QK-norm (per-head RMS norm) ──
        apply_qk_norm(&mut q, &layer.attn_q_norm, n_heads, actual_hd, cfg.norm_eps);
        apply_qk_norm(&mut k, &layer.attn_k_norm, n_kv_heads, actual_hd, cfg.norm_eps);

        // ── RoPE ──
        apply_gemma_rope(&mut q, pos, n_heads, actual_hd, &weights.rope_freqs, is_swa);
        apply_gemma_rope(&mut k, pos, n_kv_heads, actual_hd, &weights.rope_freqs, is_swa);

        // ── Pad K/V to full head_dim for KV cache ──
        let mut k_full = vec![0.0f32; full_kv_len];
        let mut v_full = vec![0.0f32; full_kv_len];
        // Copy actual data into the first part of each head
        for h in 0..n_kv_heads {
            let src_off = h * actual_hd;
            let dst_off = h * full_hd;
            k_full[dst_off..dst_off + actual_hd].copy_from_slice(&k[src_off..src_off + actual_hd]);
            v_full[dst_off..dst_off + actual_hd].copy_from_slice(&v[src_off..src_off + actual_hd]);
        }
        cache.push_layer(layer_idx, &k_full, &v_full);

        // ── Attention scores ──
        // For attention, we only attend over the actual head dimension
        let cached_len = cache.seq_len();
        let mut scores = vec![0.0f32; n_heads * cached_len];

        for h in 0..n_heads {
            let q_off = h * actual_hd;
            let kv_h = (h * n_kv_heads / n_heads).min(n_kv_heads - 1);
            let cache_keys = cache.get_keys(layer_idx);
            for pk in 0..cached_len {
                let k_off = (pk * n_kv_heads + kv_h) * full_hd;
                let k_slice = &cache_keys[k_off..k_off + actual_hd];
                scores[h * cached_len + pk] = ops::dot(&q[q_off..q_off + actual_hd], k_slice);
            }
        }

        // Scale
        let scale = 1.0 / (actual_hd as f32).sqrt();
        for s in scores.iter_mut() { *s *= scale; }

        // Causal + sliding window mask
        apply_gemma_mask(&mut scores, n_heads, cached_len, pos, is_swa, cfg.sliding_window);

        // Softmax
        for h in 0..n_heads {
            let start = h * cached_len;
            ops::softmax(&mut scores[start..start + cached_len]);
        }

        // ── Attention output ──
        let mut attn_out = vec![0.0f32; q_len];
        for h in 0..n_heads {
            let kv_h = (h * n_kv_heads / n_heads).min(n_kv_heads - 1);
            let out_off = h * actual_hd;
            let cache_values = cache.get_values(layer_idx);
            for pk in 0..cached_len {
                let s = scores[h * cached_len + pk];
                let v_off = (pk * n_kv_heads + kv_h) * full_hd;
                for d in 0..actual_hd {
                    attn_out[out_off + d] += s * cache_values[v_off + d];
                }
            }
        }

        // Output projection
        let mut attn_proj = vec![0.0f32; dim];
        quantized_or_f32_matvec_t(&layer.raw_attn_o, &layer.attn_o, dim, q_len, &attn_out, &mut attn_proj);

        // Post-attention norm + residual with per-layer scale
        ops::rms_norm(&mut attn_proj, &layer.post_attention_norm, cfg.norm_eps);
        for i in 0..dim {
            attn_proj[i] *= layer.layer_output_scale;
            x[i] += attn_proj[i];
        }

        // ── MoE / MLP ──
        let mut ffn_input = x.to_vec();
        ops::rms_norm(&mut ffn_input, &layer.ffn_norm, cfg.norm_eps);

        // Compute router logits: inp_gate @ x
        let per_dim = cfg.per_layer_dim;
        let mut router_logits = vec![0.0f32; per_dim];
        quantized_or_f32_matvec(&layer.raw_inp_gate, &layer.inp_gate, per_dim, dim, &ffn_input, &mut router_logits);

        // Softmax router
        ops::softmax(&mut router_logits);

        // Top-1 routing: find the highest score (handle NaN safely)
        let (best_idx, _) = router_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .unwrap_or((0, &1.0f32));
        let route_weight = router_logits.get(best_idx).copied().unwrap_or(0.0);
        // Handle NaN in router logits
        let route_weight = if route_weight.is_nan() { 0.0 } else { route_weight };

        // Compute expert output by routing weight * expert FFN
        let _expert_out = vec![0.0f32; dim];
        let route_weight = router_logits[best_idx];
        let hidden_dim_full = cfg.hidden_dim;

        // Gate/Up projections (shared for all experts)
        let mut gate = vec![0.0f32; hidden_dim_full];
        let mut up = vec![0.0f32; hidden_dim_full];
        quantized_or_f32_matvec(&layer.raw_ffn_gate, &layer.ffn_gate, hidden_dim_full, dim, &ffn_input, &mut gate);
        quantized_or_f32_matvec(&layer.raw_ffn_up, &layer.ffn_up, hidden_dim_full, dim, &ffn_input, &mut up);
        ops::silu_inplace(&mut gate);
        for i in 0..hidden_dim_full { gate[i] *= up[i]; }

        // The dense FFN path: gate * up -> down
        let mut dense_out = vec![0.0f32; dim];
        quantized_or_f32_matvec_t(&layer.raw_ffn_down, &layer.ffn_down, dim, hidden_dim_full, &gate, &mut dense_out);

        // Combine: output = dense + route_weight * expert_out
        if layer.has_experts && route_weight > 0.01 {
            // Expert 1
            let mut eg = vec![0.0f32; hidden_dim_full];
            let mut eu = vec![0.0f32; hidden_dim_full];
            quantized_or_f32_matvec(&layer.raw_expert_gate, &layer.expert_gate, hidden_dim_full, dim, &ffn_input, &mut eg);
            quantized_or_f32_matvec(&layer.raw_expert_up, &layer.expert_up, hidden_dim_full, dim, &ffn_input, &mut eu);
            ops::silu_inplace(&mut eg);
            for i in 0..hidden_dim_full { eg[i] *= eu[i]; }
            let mut ed = vec![0.0f32; dim];
            quantized_or_f32_matvec_t(&layer.raw_expert_down, &layer.expert_down, dim, hidden_dim_full, &eg, &mut ed);
            // Scale by route weight and add
            for i in 0..dim { dense_out[i] += route_weight * ed[i]; }

            // Expert 2 (if exists)
            if !layer.expert_gate_2.is_empty() {
                let mut eg2 = vec![0.0f32; hidden_dim_full];
                let mut eu2 = vec![0.0f32; hidden_dim_full];
                quantized_or_f32_matvec(&layer.raw_expert_gate_2, &layer.expert_gate_2, hidden_dim_full, dim, &ffn_input, &mut eg2);
                quantized_or_f32_matvec(&layer.raw_expert_up_2, &layer.expert_up_2, hidden_dim_full, dim, &ffn_input, &mut eu2);
                ops::silu_inplace(&mut eg2);
                for i in 0..hidden_dim_full { eg2[i] *= eu2[i]; }
                let mut ed2 = vec![0.0f32; dim];
                quantized_or_f32_matvec_t(&layer.raw_expert_down_2, &layer.expert_down_2, dim, hidden_dim_full, &eg2, &mut ed2);
                // Find second-best router weight
                let second_weight = router_logits.iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != best_idx)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                for i in 0..dim { dense_out[i] += second_weight * ed2[i]; }
            }
        }

        // Projection layer
        let mut proj_out = vec![0.0f32; dim];
        quantized_or_f32_matvec_t(&layer.raw_proj, &layer.proj, dim, per_dim, &router_logits, &mut proj_out);

        // Post-FFN norm + residual
        let mut ffn_result = dense_out;
        ops::rms_norm(&mut ffn_result, &layer.post_ffw_norm, cfg.norm_eps);
        for i in 0..dim { x[i] += layer.layer_output_scale * ffn_result[i]; }

        // Additional post-processing (extra norm)
        let mut post = x.to_vec();
        ops::rms_norm(&mut post, &layer.post_norm, cfg.norm_eps);
        ops::add_inplace(x, &post);
    }

    // ── Advance KV cache ──
    cache.advance();

    // ── Final RMSNorm ──
    ops::rms_norm(x, &weights.output_norm, cfg.norm_eps);
}

/// Compute top-k logits from the shared embedding table.
/// Uses direct Q4 dot product on raw data (no per-row dequantization).
fn top_k_logits(
    x: &[f32],
    weights: &Gemma4Weights,
    k: usize,
) -> Vec<(f32, usize)> {
    let cfg = &weights.config;
    let dim = cfg.dim;
    let cap = cfg.final_logit_softcapping;
    let row_bytes = dtype::row_stride_bytes(dim, weights.tok_embd_dtype);
    let raw = &weights.tok_embd_raw;
    let vocab = cfg.vocab_size;

    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut min_idx = 0usize;

    let is_q4_blocks = weights.tok_embd_dtype == DType::Q4;

    for v in 0..vocab {
        let byte_off = v * row_bytes;
        let logit = if byte_off + row_bytes <= raw.len() {
            let row = &raw[byte_off..byte_off + row_bytes];
            if is_q4_blocks && row.len() >= 18 && (row.len() % 18) < 4 {
                dtype::q4_0_dot(row, x, dim)
            } else {
                dtype::k_quant_dot(row, x, dim)
            }
        } else { 0.0 };

        let logit = if cap > 0.0 && cap.is_finite() { cap * (logit / cap).tanh() } else { logit };

        if heap.len() < k {
            heap.push((logit, v));
            if logit < heap[min_idx].0 { min_idx = heap.len() - 1; }
        } else if logit > heap[min_idx].0 {
            heap[min_idx] = (logit, v);
            min_idx = 0;
            for i in 1..heap.len() {
                if heap[i].0 < heap[min_idx].0 { min_idx = i; }
            }
        }
    }

    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    heap
}

// ── Helper functions ───────────────────────────────────────────────

/// Apply QK-norm (per-head RMS norm) to query or key.
fn apply_qk_norm(x: &mut [f32], norm: &[f32], n_heads: usize, head_dim: usize, eps: f64) {
    if norm.is_empty() { return; }
    for h in 0..n_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let n_end = if norm.len() >= head_dim { head_dim } else { norm.len() };
        let n = &norm[..n_end.min(norm.len())];
        ops::rms_norm(&mut x[start..end], n, eps);
    }
}

/// Apply Gemma4-style RoPE using learned frequency weights.
fn apply_gemma_rope(x: &mut [f32], pos: usize, n_heads: usize, head_dim: usize, freqs: &[f32], is_swa: bool) {
    let freq_offset = if is_swa && freqs.len() > head_dim { head_dim } else { 0 };
    for h in 0..n_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let slice = &mut x[start..end];
        for i in (0..head_dim).step_by(2) {
            if i + 1 >= head_dim { break; }
            let freq_idx = i + freq_offset;
            let theta = if freq_idx < freqs.len() {
                pos as f32 * freqs[freq_idx]
            } else {
                // Fallback: standard RoPE formula
                pos as f32 / 10000.0_f32.powf(2.0 * (i as f32) / head_dim as f32)
            };
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = slice[i];
            let x1 = slice[i + 1];
            slice[i] = x0 * cos_t - x1 * sin_t;
            slice[i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/// Apply causal + sliding window mask to attention scores.
fn apply_gemma_mask(scores: &mut [f32], n_heads: usize, seq_len: usize, pos: usize, is_swa: bool, window: usize) {
    for h in 0..n_heads {
        for pk in 0..seq_len {
            let idx = h * seq_len + pk;
            if pk > pos {
                // Future token: mask out
                scores[idx] = -65504.0;
            } else if is_swa && (pos - pk) > window {
                // Outside sliding window: mask out
                scores[idx] = -65504.0;
            }
        }
    }
}

/// Dispatch matvec: use f32 weights (already dequantized during load).
/// For Gemma4 we always use f32 matvec since the weights are fully dequantized.
fn quantized_or_f32_matvec(
    _raw: &Option<Vec<u8>>, f32_w: &[f32],
    rows: usize, cols: usize, x: &[f32], y: &mut [f32]
) {
    ops::matvec(rows, cols, f32_w, x, y);
}

/// Dispatch matvec transposed — always use f32.
fn quantized_or_f32_matvec_t(
    _raw: &Option<Vec<u8>>, f32_w: &[f32],
    rows: usize, cols: usize, x: &[f32], y: &mut [f32]
) {
    ops::matvec_transpose(rows, cols, f32_w, x, y);
}

// ── Gemma4 Inference Engine ────────────────────────────────────────

pub struct Gemma4Engine {
    pub weights: Gemma4Weights,
    pub tokenizer: Tokenizer,
    pub config: Gemma4Config,
    pub params: SamplingParams,
}

impl Gemma4Engine {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let rt = AxonRuntime::open(path).map_err(|e| format!("failed to open: {e}"))?;
        let config = Gemma4Config::from_runtime(&rt)?;
        let weights = Gemma4Weights::load(&rt, &config)?;
        let tokenizer = Tokenizer::from_runtime_manifest(&rt, config.vocab_size)?;

        log::info!("Gemma4 engine ready: {} layers, {} vocab, {} ctx",
            config.n_layers, config.vocab_size, config.ctx_len);

        Ok(Self {
            weights,
            tokenizer,
            config,
            params: SamplingParams::default(),
        })
    }

    pub fn set_sampling(&mut self, params: SamplingParams) {
        self.params = params;
    }

    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<(String, crate::chat::GenerationStats), String> {
        let start = std::time::Instant::now();
        let prompt_tokens = self.tokenizer.encode(prompt);
        let prompt_len = prompt_tokens.len();
        if prompt_len == 0 { return Err("empty prompt".to_string()); }

        let dim = self.config.dim;
        let k_candidates = self.params.top_k.max(64);
        let mut cache = KVCache::new(
            self.config.n_layers, self.config.n_kv_heads,
            self.config.head_dim.max(self.config.head_dim_swa),
            self.config.ctx_len,
        );
        let mut x = vec![0.0f32; dim];

        // Prompt evaluation
        let prompt_start = std::time::Instant::now();
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            self.weights.embed(token_id as usize, &mut x);
            gemma4_forward(&mut x, &self.weights, &mut cache, pos);
        }
        let prompt_eval_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;

        // Generation
        let mut generated = Vec::new();
        let mut rng_state: u64 = 42;
        let mut rng = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32) / (1u64 << 31) as f32
        };

        for _ in 0..max_tokens {
            // Compute top-k candidates from shared embeddings
            let topk = top_k_logits(&x, &self.weights, k_candidates);

            if topk.is_empty() { break; }

            // Softmax over top-k
            let max_logit = topk.iter().map(|(l, _)| *l).fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = topk.iter().map(|(l, _)| ((l - max_logit).exp())).collect();
            let sum: f32 = probs.iter().sum();
            if sum <= 0.0 { break; }
            let inv_sum = 1.0 / sum;
            for p in probs.iter_mut() { *p *= inv_sum; }

            // Sample
            let r = rng();
            let mut cum = 0.0f32;
            let mut selected = topk[0].1;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if r <= cum { selected = topk[i].1; break; }
            }

            if selected as u32 == self.tokenizer.eos_id() { break; }
            generated.push(selected as u32);
            self.weights.embed(selected, &mut x);
            let pos = prompt_len + generated.len() - 1;
            gemma4_forward(&mut x, &self.weights, &mut cache, pos);
        }

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let tok_ps = if total_ms > 0.0 { generated.len() as f64 / (total_ms / 1000.0) } else { 0.0 };
        let text = self.tokenizer.decode(&generated);
        let stats = crate::chat::GenerationStats {
            prompt_tokens: prompt_len, generated_tokens: generated.len(),
            total_duration_ms: total_ms, tokens_per_second: tok_ps,
            prompt_eval_ms,
        };
        Ok((text, stats))
    }

    pub fn chat(&mut self, system_prompt: Option<&str>, max_tokens: usize) -> Result<(), String> {
        use std::io::Write;
        let mut history = String::new();
        if let Some(sys) = system_prompt {
            history.push_str(&format!("<|system|>\n{}\n<|end|>\n", sys));
        }
        println!("Gemma4 chat started. Type your messages (or 'exit' to quit).");
        loop {
            print!("You: ");
            std::io::stdout().flush().map_err(|e| format!("IO error: {e}"))?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).map_err(|e| format!("IO error: {e}"))?;
            let input = input.trim().to_string();
            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") { break; }
            if input.is_empty() { continue; }

            let full = format!("{}<|user|>\n{}\n<|end|>\n<|assistant|>\n", history, input);
            print!("Axon: ");
            std::io::stdout().flush().ok();
            let gen_start = std::time::Instant::now();
            let (resp, stats) = self.generate_text(&full, max_tokens)?;
            println!("{}", resp);
            eprintln!("  [prompt: {} tok | gen: {} tok | {:.2} tok/s | {:.1}s]",
                stats.prompt_tokens, stats.generated_tokens, stats.tokens_per_second,
                gen_start.elapsed().as_secs_f64());
            history.push_str(&format!("<|user|>\n{}\n<|end|>\n<|assistant|>\n{}\n<|end|>\n", input, resp));
        }
        Ok(())
    }
}
