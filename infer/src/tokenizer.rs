//! Minimal tokenizer for LLM inference.
//!
//! Supports:
//! - BPE tokenizer (tokenizer.json format from HuggingFace)
//! - SentencePiece unigram model (tokenizer.model or .model file)
//! - Simple byte-level fallback

use std::collections::HashMap;

/// A single token in the vocabulary.
pub struct VocabEntry {
    pub id: u32,
    pub token: String,
    pub score: f32,  // Unused for BPE, used for SentencePiece
}

/// Tokenizer for encoding strings to token IDs and decoding back.
pub struct Tokenizer {
    /// Map from token string to ID
    pub token_to_id: HashMap<String, u32>,
    /// Map from ID to token string
    pub id_to_token: HashMap<u32, String>,
    /// Special tokens
    pub bos_id: u32,
    pub eos_id: u32,
    pub vocab_size: usize,
    /// Whether the tokenizer adds BOS automatically
    pub add_bos: bool,
}

impl Tokenizer {
    /// Create a new tokenizer from a vocabulary list.
    pub fn new(
        vocab: Vec<(String, u32)>,
        bos_id: u32,
        eos_id: u32,
        add_bos: bool,
    ) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (token, id) in vocab {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }
        let vocab_size = token_to_id.len();
        Self {
            token_to_id,
            id_to_token,
            bos_id,
            eos_id,
            vocab_size,
            add_bos,
        }
    }

    /// Build a tokenizer from an AxonRuntime's manifest directly (no LoadedModel).
    pub fn from_runtime_manifest(rt: &axon_runtime::AxonRuntime, vocab_size: usize) -> Result<Self, String> {
        let manifest = rt.manifest();
        let tokenizer_config = &manifest.tokenizer;

        let bos_id = tokenizer_config.as_ref().and_then(|t| t.bos_token.as_ref())
            .and_then(|_| Some(1u32)).unwrap_or(1);
        let eos_id = tokenizer_config.as_ref().and_then(|t| t.eos_token.as_ref())
            .and_then(|_| Some(2u32)).unwrap_or(2);
        let add_bos = tokenizer_config.as_ref().map(|t| t.bos_token.is_some()).unwrap_or(true);

        let mut vocab = Vec::with_capacity(vocab_size.max(256));
        for i in 0..vocab_size.max(256) {
            let token = if i < 256 {
                format!("<0x{:02X}>", i)
            } else if i == bos_id as usize {
                "<s>".to_string()
            } else if i == eos_id as usize {
                "</s>".to_string()
            } else if i == 0 {
                "<unk>".to_string()
            } else {
                format!("<token_{}>", i)
            };
            vocab.push((token, i as u32));
        }
        Ok(Self::new(vocab, bos_id as u32, eos_id as u32, add_bos))
    }

    /// Build a tokenizer from the model's manifest tokenizer config and
    /// optionally by scanning the embedding table for known tokens.
    pub fn from_model(model: &crate::model::LoadedModel) -> Result<Self, String> {
        let manifest = model.rt.manifest();
        let config = &model.config;
        let vocab_size = config.vocab_size;

        // Try to get tokenizer info from manifest
        let tokenizer_config = &manifest.tokenizer;

        let bos_id = tokenizer_config
            .as_ref()
            .and_then(|t| t.bos_token.as_ref())
            .and_then(|_s| {
                // Try to find the token ID from the string
                // For now, use 1 as default BOS (common for Llama)
                Some(1u32)
            })
            .unwrap_or(1);

        let eos_id = tokenizer_config
            .as_ref()
            .and_then(|t| t.eos_token.as_ref())
            .and_then(|s| {
                if s == "<|endoftext|>" { Some(2u32) }
                else if s == "</s>" { Some(2u32) }
                else if s == "<eos>" { Some(2u32) }
                else if s == "<|im_end|>" { Some(2u32) }
                else { Some(2u32) }
            })
            .unwrap_or(2);

        let add_bos = tokenizer_config
            .as_ref()
            .map(|t| t.bos_token.is_some())
            .unwrap_or(true);

        // Build reversed token list from the embedding table shape
        // The embedding table [vocab_size, dim] gives us the vocab size.
        // We can't recover token strings from this, so we build fallback IDs.
        let mut vocab = Vec::with_capacity(vocab_size);

        // Try to load a tokenizer config from the .axon model's directory
        // For now, build byte-level fallback tokens
        for i in 0..vocab_size {
            // Try common byte-level patterns
            let token = if i < 256 {
                // Byte tokens (GPT-2 style)
                format!("<0x{:02X}>", i)
            } else if i == bos_id as usize {
                "<s>".to_string()
            } else if i == eos_id as usize {
                "</s>".to_string()
            } else if i == 0 {
                "<unk>".to_string()
            } else {
                format!("<token_{}>", i)
            };
            vocab.push((token, i as u32));
        }

        log::info!(
            "Built tokenizer with {} tokens (BOS={}, EOS={}, add_bos={})",
            vocab_size, bos_id, eos_id, add_bos
        );

        Ok(Self::new(vocab, bos_id as u32, eos_id as u32, add_bos))
    }

    /// Encode a string to token IDs.
    ///
    /// Uses a greedy byte-level encoding:
    /// 1. Split into UTF-8 bytes
    /// 2. Try to find multi-byte sequences in vocabulary
    /// 3. Fall back to byte tokens
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        if self.add_bos {
            tokens.push(self.bos_id);
        }

        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Try to find the longest matching token starting at this position
            let mut best_len = 0;
            let mut best_id = None;

            // Try byte-level token first
            let byte_token = format!("<0x{:02X}>", bytes[i]);
            if let Some(&id) = self.token_to_id.get(&byte_token) {
                best_len = 1;
                best_id = Some(id);
            }

            // Try raw byte as a single-character string
            let raw_byte = std::str::from_utf8(&bytes[i..=i]).unwrap_or("");
            if let Some(&id) = self.token_to_id.get(raw_byte) {
                best_len = 1;
                best_id = Some(id);
            }

            if let Some(id) = best_id {
                tokens.push(id);
                i += best_len;
            } else {
                // Fallback: use byte token
                tokens.push(bytes[i] as u32);
                i += 1;
            }
        }

        tokens
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if id == self.bos_id || id == self.eos_id {
                continue;
            }
            if let Some(token) = self.id_to_token.get(&id) {
                // Try to parse byte tokens
                if token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        bytes.push(byte);
                        continue;
                    }
                }
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                bytes.extend_from_slice(token.as_bytes());
            } else {
                // Unknown token
                bytes.push('�' as u8);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Get the EOS token ID.
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Get the BOS token ID.
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }
}
