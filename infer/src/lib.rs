//! # axon-infer
//!
//! Native inference engine for `.axon` model files.
//!
//! Loads models converted from GGUF/Ollama and runs them directly
//! using CPU inference with quantized matmul support.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use axon_infer::chat::InferenceEngine;
//! use axon_infer::sampling::SamplingParams;
//! use std::path::Path;
//!
//! let mut engine = InferenceEngine::load(Path::new("model.axon")).unwrap();
//! engine.set_sampling(SamplingParams {
//!     temperature: 0.7,
//!     top_k: 40,
//!     top_p: 0.9,
//!     repeat_penalty: 1.1,
//! });
//! let (response, stats) = engine.generate_text("Hello!", 256).unwrap();
//! println!("{}", response);
//! ```

pub mod chat;
pub mod dtype;
pub mod gemma4;
pub mod kv_cache;
pub mod model;
pub mod ops;
pub mod quantized;
pub mod sampling;
pub mod tokenizer;
pub mod transformer;
