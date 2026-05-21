//! Convert between .axon and other model weight formats.
//!
//! Supports:
//! - **SafeTensors import**: read .safetensors files into .axon
//! - **JSON export**: dump manifest as human-readable JSON
//! - **Raw tensor I/O**: write individual weight blobs to disk

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{AxonError, AxonResult};
use crate::manifest::AxonFile;
use crate::mmap_loader::AxonBuilder;
use crate::tensor::DType;

// ── SafeTensors import ────────────────────────────────────────────

#[derive(Debug)]
pub struct SafeTensorsHeader {
    pub metadata: HashMap<String, serde_json::Value>,
    pub tensors: Vec<SafeTensorEntry>,
}

#[derive(Debug)]
pub struct SafeTensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data_offset: u64,
    pub data_size: u64,
}

/// Parse a .safetensors header from raw bytes.
///
/// The header is a JSON block preceded by an 8-byte little-endian length.
pub fn parse_safetensors_header(data: &[u8]) -> AxonResult<SafeTensorsHeader> {
    if data.len() < 8 {
        return Err(AxonError::UnexpectedEof {
            needed: 8,
            available: data.len() as u64,
        });
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > data.len() {
        return Err(AxonError::UnexpectedEof {
            needed: (8 + header_len) as u64,
            available: data.len() as u64,
        });
    }

    // The actual header size includes padding to make (8 + header_len) % 64 == 0
    // But the header_len says exactly how many JSON bytes to read.
    let header_json: serde_json::Value = serde_json::from_slice(&data[8..8 + header_len])
        .map_err(|e| AxonError::InvalidManifest(format!("SafeTensors header JSON: {e}")))?;

    let mut metadata = HashMap::new();
    let mut tensors = Vec::new();

    if let Some(obj) = header_json.as_object() {
        for (name, value) in obj {
            if name == "__metadata__" {
                // Store metadata
                if let Some(meta_obj) = value.as_object() {
                    for (k, v) in meta_obj {
                        metadata.insert(k.clone(), v.clone());
                    }
                }
                continue;
            }
            if let Some(entry) = value.as_object() {
                match convert_safetensor_entry(name, entry, 8 + header_len) {
                    Ok(t) => tensors.push(t),
                    Err(e) => log::warn!("Skipping tensor '{name}': {e}"),
                }
            }
        }
    }

    tensors.sort_by_key(|t| t.data_offset);
    Ok(SafeTensorsHeader { metadata, tensors })
}

fn convert_safetensor_entry(
    name: &str,
    entry: &serde_json::Map<String, serde_json::Value>,
    payload_base: usize,
) -> AxonResult<SafeTensorEntry> {
    let dtype_str = entry
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AxonError::InvalidManifest(format!("{name}: missing dtype")))?;

    let shape: Vec<u64> = entry
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AxonError::InvalidManifest(format!("{name}: missing shape")))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(1))
        .collect();

    let data_offsets = entry
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AxonError::InvalidManifest(format!("{name}: missing data_offsets")))?;

    let begin = data_offsets.first().and_then(|v| v.as_u64()).unwrap_or(0);
    let end = data_offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(0);

    let dtype = match dtype_str {
        "F32" | "float32" => DType::F32,
        "F16" | "float16" | "half" => DType::F16,
        "BF16" | "bfloat16" => DType::BF16,
        "I32" | "int32" => DType::I32,
        "I64" | "int64" => DType::I64,
        "I8" | "int8" => DType::I8,
        "U8" | "uint8" => DType::U8,
        _ => return Err(AxonError::InvalidDtype(0)),
    };

    Ok(SafeTensorEntry {
        name: name.to_string(),
        dtype,
        shape,
        data_offset: payload_base as u64 + begin,
        data_size: end - begin,
    })
}

/// Convert an entire .safetensors file to .axon format in-memory.
pub fn safetensors_to_axon(path: &Path) -> AxonResult<Vec<u8>> {
    let data = fs::read(path)?;
    let header = parse_safetensors_header(&data)?;
    let mut builder = AxonBuilder::new();

    for entry in &header.tensors {
        let start = entry.data_offset as usize;
        let end = start + entry.data_size as usize;
        if end > data.len() {
            return Err(AxonError::UnexpectedEof {
                needed: end as u64,
                available: data.len() as u64,
            });
        }
        let tensor_bytes = data[start..end].to_vec();
        builder = builder.add_tensor(&entry.name, tensor_bytes, entry.dtype, &entry.shape);
    }

    builder.build()
}

// ── JSON / raw I/O ────────────────────────────────────────────────

/// Export an .axon file's manifest as pretty-printed JSON.
pub fn export_manifest_json(axon_data: &[u8]) -> AxonResult<String> {
    let file = AxonFile::from_bytes(axon_data.to_vec())?;
    let manifest = &file.manifest;

    let tensors: Vec<serde_json::Value> = manifest
        .tensor_order
        .iter()
        .map(|name| {
            let desc = manifest.get_tensor(name).unwrap();
            let dtype_name = desc.dtype().map(|d| d.name()).unwrap_or("UNKNOWN");
            serde_json::json!({
                "name": name,
                "dtype": dtype_name,
                "shape": desc.shape_vec(),
                "size_bytes": desc.data_size,
            })
        })
        .collect();

    let output = serde_json::json!({
        "model": manifest.model,
        "architecture": manifest.architecture,
        "hyperparameters": manifest.hyperparameters,
        "tokenizer": manifest.tokenizer,
        "context_length": manifest.context_length,
        "quantization": manifest.quantization,
        "tensor_count": manifest.tensor_count(),
        "tensors": tensors,
    });

    Ok(serde_json::to_string_pretty(&output)?)
}

/// Write all tensors from an .axon file as individual raw binary blobs.
pub fn export_tensors_raw(axon_data: &[u8], output_dir: &Path) -> AxonResult<u64> {
    let file = AxonFile::from_bytes(axon_data.to_vec())?;
    fs::create_dir_all(output_dir)?;

    let mut count = 0u64;
    for name in &file.manifest.tensor_order {
        if let Some(tensor_data) = file.tensor_data(name) {
            let safe_name = name.replace('/', "_");
            let out_path = output_dir.join(format!("{safe_name}.bin"));
            fs::write(&out_path, tensor_data)?;
            count += 1;
        }
    }
    Ok(count)
}
