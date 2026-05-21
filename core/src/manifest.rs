use crate::error::{AxonError, AxonResult};
use crate::header::AxonHeader;
use crate::tensor::TensorDescriptor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub model: Option<String>,
    pub architecture: Option<String>,
    #[serde(default)]
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub tokenizer: Option<TokenizerConfig>,
    pub context_length: Option<u64>,
    #[serde(skip)]
    pub tensors: HashMap<String, TensorDescriptor>,
    pub tensor_order: Vec<String>,
    #[serde(default)]
    pub patches: Vec<PatchInfo>,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub model_type: String,
    pub vocab_size: u64,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub method: String,
    pub group_size: Option<u64>,
    pub bits: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchInfo {
    pub name: String,
    pub file: String,
    pub description: Option<String>,
}

impl Manifest {
    pub fn new() -> Self {
        Self {
            model: None,
            architecture: None,
            hyperparameters: HashMap::new(),
            tokenizer: None,
            context_length: None,
            tensors: HashMap::new(),
            tensor_order: Vec::new(),
            patches: Vec::new(),
            quantization: None,
        }
    }

    pub fn add_tensor(&mut self, desc: TensorDescriptor) {
        let name = desc.name_str().to_string();
        self.tensor_order.push(name.clone());
        self.tensors.insert(name, desc);
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    pub fn to_json_bytes(&self) -> AxonResult<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(self)?)
    }
    pub fn from_json_bytes(bytes: &[u8]) -> AxonResult<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }
    pub fn tensor_count(&self) -> u64 {
        self.tensors.len() as u64
    }

    pub fn validate(&self) -> AxonResult<()> {
        if self.tensors.is_empty() {
            return Err(AxonError::InvalidManifest(
                "Manifest must contain at least one tensor".into(),
            ));
        }
        Ok(())
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AxonFile {
    pub header: AxonHeader,
    pub manifest: Manifest,
    pub data: Vec<u8>,
}

impl AxonFile {
    pub fn from_bytes(data: Vec<u8>) -> AxonResult<Self> {
        let header = AxonHeader::from_bytes(&data)?;
        let manifest_start = header.manifest_offset as usize;
        let manifest_end = manifest_start + header.manifest_size as usize;
        if manifest_end > data.len() {
            return Err(AxonError::UnexpectedEof {
                needed: manifest_end as u64,
                available: data.len() as u64,
            });
        }
        let mut manifest = Manifest::from_json_bytes(&data[manifest_start..manifest_end])?;
        let tdt_start = AxonHeader::align_up(header.manifest_offset + header.manifest_size, 64);
        let tdt_end = tdt_start + (header.tensor_count * TensorDescriptor::SIZE as u64);
        if (tdt_end as usize) <= data.len() {
            let mut cursor = tdt_start as usize;
            for _ in 0..header.tensor_count {
                if cursor + TensorDescriptor::SIZE > data.len() {
                    break;
                }
                let desc = TensorDescriptor::from_bytes(&data[cursor..])?;
                cursor += TensorDescriptor::SIZE;
                manifest.add_tensor(desc);
            }
        }
        Ok(Self {
            header,
            manifest,
            data,
        })
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let desc = self.manifest.get_tensor(name)?;
        let start = desc.data_offset as usize;
        let end = start + desc.data_size as usize;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    pub fn verify_all_checksums(&self) -> Vec<(String, bool)> {
        let mut results = Vec::new();
        for (name, desc) in &self.manifest.tensors {
            if desc.checksum != 0 {
                if let Some(data) = self.tensor_data(name) {
                    results.push((
                        name.clone(),
                        crate::checksum::verify_checksum(data, desc.checksum),
                    ));
                } else {
                    results.push((name.clone(), false));
                }
            }
        }
        results
    }
}
