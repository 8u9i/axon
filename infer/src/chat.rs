//! Interactive chat loop for model inference.
//!
//! Manages conversation history, token generation, and interactive I/O.

use std::time::Instant;

use axon_core::DType;

use crate::dtype;
use crate::kv_cache::KVCache;
use crate::model::LoadedModel;
use crate::ops;
use crate::sampling::{sample, SamplingParams};
use crate::tokenizer::Tokenizer;
use crate::transformer::{forward, ModelWeights};

/// A single message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,  // "user", "assistant", "system"
    pub content: String,
}

/// Generation stats.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_duration_ms: f64,
    pub tokens_per_second: f64,
    pub prompt_eval_ms: f64,
}

/// The inference engine.
pub struct InferenceEngine {
    pub model: LoadedModel,
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    pub config: crate::model::ModelConfig,
    pub params: SamplingParams,
}

impl InferenceEngine {
    /// Load a model and prepare it for inference.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let model = LoadedModel::open(path)?;
        let config = model.config.clone();
        let tokenizer = Tokenizer::from_model(&model)?;
        let weights = ModelWeights::load(&model)?;

        Ok(Self {
            model,
            weights,
            tokenizer,
            config,
            params: SamplingParams::default(),
        })
    }

    /// Set sampling parameters.
    pub fn set_sampling(&mut self, params: SamplingParams) {
        self.params = params;
    }

    /// Generate tokens from a prompt.
    ///
    /// Returns generated tokens and stats.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, GenerationStats), String> {
        let config = &self.config;
        let start = Instant::now();

        // Encode prompt
        let prompt_tokens = self.tokenizer.encode(prompt);
        let prompt_len = prompt_tokens.len();
        if prompt_len == 0 {
            return Err("empty prompt".to_string());
        }

        log::info!(
            "Generating with {} prompt tokens, max {} new tokens",
            prompt_len,
            max_tokens
        );

        // Create KV cache
        let mut cache = KVCache::new(
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            config.ctx_len,
        );

        // Embedding dimension
        let dim = config.dim;
        let vocab = config.vocab_size;

        // Pre-allocate buffers
        let mut x = vec![0.0f32; dim];
        let mut logits = vec![0.0f32; vocab];

        // --- Prompt evaluation: process all prompt tokens sequentially ---
        let prompt_eval_start = Instant::now();

        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            // Look up embedding
            let emb_start = (token_id as usize) * dim;

            if self.weights.embed_dtype == DType::F32 || self.weights.embed_dtype == DType::F16 {
                // Already have f32 embeddings loaded
                if emb_start + dim <= self.weights.tok_embeddings.len() {
                    ops::copy(&mut x, &self.weights.tok_embeddings[emb_start..emb_start + dim]);
                } else {
                    log::warn!("Token {} out of embedding range (size {})", token_id, self.weights.tok_embeddings.len() / dim);
                    x.fill(0.0);
                }
            } else if let Some(ref raw) = self.weights.raw_tok_embeddings {
                // Need to dequantize
                let dtype = self.weights.embed_dtype;
                let row_bytes = dtype::row_stride_bytes(dim, dtype);
                let byte_start = emb_start / dim * row_bytes;
                if byte_start + row_bytes <= raw.len() {
                    let row_data = &raw[byte_start..byte_start + row_bytes];
                    let deq = dtype::dequantize_tensor(row_data, dtype);
                    ops::copy(&mut x, &deq[..dim.min(deq.len())]);
                }
            } else {
                x.fill(0.0);
            }

            // Forward pass
            forward(&mut x, &mut logits, &self.weights, &mut cache, pos);
        }

        let prompt_eval_time = prompt_eval_start.elapsed().as_secs_f64() * 1000.0;

        // --- Token generation ---
        let mut generated_tokens = Vec::new();
        let mut rng_state: u64 = 42;
        let mut rng = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32) / (1u64 << 31) as f32
        };

        for _ in 0..max_tokens {
            // Sample next token
            let next_token = sample(&logits, &self.params, &mut rng);

            // Check for EOS
            if next_token == self.tokenizer.eos_id() as usize {
                break;
            }

            generated_tokens.push(next_token as u32);

            // Prepare input for next token (embedding lookup)
            let emb_start = (next_token as usize) * dim;
            if emb_start + dim <= self.weights.tok_embeddings.len() {
                ops::copy(&mut x, &self.weights.tok_embeddings[emb_start..emb_start + dim]);
            } else {
                x.fill(0.0);
            }

            // Forward pass for the new token
            let pos = prompt_len + generated_tokens.len() - 1;
            forward(&mut x, &mut logits, &self.weights, &mut cache, pos);
        }

        let total_duration = start.elapsed().as_secs_f64() * 1000.0;
        let total_tokens = generated_tokens.len();
        let tok_per_sec = if total_duration > 0.0 {
            total_tokens as f64 / (total_duration / 1000.0)
        } else {
            0.0
        };

        let stats = GenerationStats {
            prompt_tokens: prompt_len,
            generated_tokens: total_tokens,
            total_duration_ms: total_duration,
            tokens_per_second: tok_per_sec,
            prompt_eval_ms: prompt_eval_time,
        };

        Ok((generated_tokens, stats))
    }

    /// Generate a response text from a prompt.
    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(String, GenerationStats), String> {
        let (tokens, stats) = self.generate(prompt, max_tokens)?;
        let text = self.tokenizer.decode(&tokens);
        Ok((text, stats))
    }

    /// Run an interactive chat session.
    pub fn chat(
        &mut self,
        system_prompt: Option<&str>,
        max_tokens: usize,
    ) -> Result<(), String> {
        let mut history = String::new();

        // Apply system prompt
        if let Some(sys) = system_prompt {
            history.push_str(&format!("<|system|>\n{}\n<|end|>\n", sys));
        }

        println!("Chat started. Type your messages (or 'exit' to quit).");
        if system_prompt.is_some() {
            println!("System prompt active.");
        }
        println!();

        loop {
            print!("You: ");
            use std::io::Write;
            std::io::stdout().flush().map_err(|e| format!("IO error: {e}"))?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).map_err(|e| format!("IO error: {e}"))?;
            let input = input.trim().to_string();

            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("/exit") {
                break;
            }
            if input.is_empty() {
                continue;
            }

            // Build prompt with history
            let full_prompt = format!(
                "{}<|user|>\n{}\n<|end|>\n<|assistant|>\n",
                history, input
            );

            print!("Axon: ");
            std::io::stdout().flush().ok();

            let gen_start = Instant::now();
            let (response, stats) = self.generate_text(&full_prompt, max_tokens)?;
            let gen_time = gen_start.elapsed().as_secs_f64();

            println!("{}", response);

            // Show stats
            print_stats(&stats, gen_time);

            // Update history
            history.push_str(&format!(
                "<|user|>\n{}\n<|end|>\n<|assistant|>\n{}\n<|end|>\n",
                input, response
            ));

            // Trim history if too long
            if history.len() > self.config.ctx_len * 4 {
                let half = history.len() / 2;
                if let Some(pos) = history[half..].find("<|user|>") {
                    history = history[half + pos..].to_string();
                }
            }
        }

        Ok(())
    }

    /// Get a reference to the model configuration.
    pub fn config(&self) -> &crate::model::ModelConfig {
        &self.config
    }
}

fn print_stats(stats: &GenerationStats, wall_secs: f64) {
    eprintln!(
        "  [prompt: {} tok | gen: {} tok | {:.2} tok/s | total: {:.1}s]",
        stats.prompt_tokens,
        stats.generated_tokens,
        stats.tokens_per_second,
        wall_secs
    );
}
