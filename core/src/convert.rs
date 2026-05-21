//! Convert between .axon and other model weight formats.
//!
//! Supports:
//! - **SafeTensors import**: read .safetensors files into .axon
//! - **JSON export**: dump manifest as human-readable JSON
//! - **Raw tensor I/O**: write individual weight blobs to disk

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub tensors: Vec<GgufTensorEntry>,
}

#[derive(Debug, Clone)]
pub struct GgufTensorEntry {
    pub name: String,
    pub ggml_type: u32,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data_offset: u64,
    pub data_size: u64,
}

#[derive(Debug, Clone)]
pub struct OllamaModelSource {
    pub manifest_path: PathBuf,
    pub blob_path: PathBuf,
    pub model_ref: String,
    pub digest: String,
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

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

/// Parse a GGUF v2/v3 file header, metadata, and tensor directory.
///
/// Tensor bytes are not interpreted or dequantized. The importer copies the
/// stored GGUF tensor byte ranges into Axon so they can be inspected, validated,
/// and accessed lazily by Axon tooling.
pub fn parse_gguf_header(data: &[u8]) -> AxonResult<GgufHeader> {
    let mut cursor = Cursor::new(data);
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(AxonError::InvalidManifest(format!(
            "invalid GGUF magic: expected GGUF, got {magic:?}"
        )));
    }

    let version = read_u32(&mut cursor)?;
    if !(2..=3).contains(&version) {
        return Err(AxonError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(&mut cursor)?;
    let metadata_kv_count = read_u64(&mut cursor)?;
    let mut metadata = HashMap::new();

    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let value_type = read_u32(&mut cursor)?;
        let value = read_gguf_value(&mut cursor, value_type)?;
        metadata.insert(key, value);
    }

    let mut tensor_infos = Vec::new();
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut cursor)?;
        let n_dims = read_u32(&mut cursor)?;
        if n_dims == 0 || n_dims as usize > crate::tensor::MAX_TENSOR_RANK {
            return Err(AxonError::InvalidManifest(format!(
                "GGUF tensor {name} has unsupported rank {n_dims}"
            )));
        }

        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut cursor)?);
        }
        let ggml_type = read_u32(&mut cursor)?;
        let relative_offset = read_u64(&mut cursor)?;
        let dtype = ggml_type_to_dtype(ggml_type);
        tensor_infos.push((name, ggml_type, dtype, shape, relative_offset));
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32)
        .max(1);
    let data_start = align_up(cursor.position(), alignment);
    let mut sorted_offsets: Vec<u64> = tensor_infos
        .iter()
        .map(|(_, _, _, _, offset)| data_start + *offset)
        .collect();
    sorted_offsets.sort_unstable();
    sorted_offsets.dedup();

    let file_len = data.len() as u64;
    let tensors = tensor_infos
        .into_iter()
        .map(|(name, ggml_type, dtype, shape, relative_offset)| {
            let data_offset = data_start + relative_offset;
            let next_offset = sorted_offsets
                .iter()
                .copied()
                .find(|offset| *offset > data_offset)
                .unwrap_or(file_len);
            if data_offset > file_len || next_offset > file_len || next_offset < data_offset {
                return Err(AxonError::UnexpectedEof {
                    needed: next_offset,
                    available: file_len,
                });
            }
            Ok(GgufTensorEntry {
                name,
                ggml_type,
                dtype,
                shape,
                data_offset,
                data_size: next_offset - data_offset,
            })
        })
        .collect::<AxonResult<Vec<_>>>()?;

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata,
        tensors,
    })
}

/// Convert a GGUF file into Axon format in memory.
pub fn gguf_to_axon(path: &Path) -> AxonResult<Vec<u8>> {
    gguf_to_axon_with_metadata(path, HashMap::new())
}

fn gguf_to_axon_with_metadata(
    path: &Path,
    extra_metadata: HashMap<String, serde_json::Value>,
) -> AxonResult<Vec<u8>> {
    let data = fs::read(path)?;
    let header = parse_gguf_header(&data)?;

    let model = header
        .metadata
        .get("ollama.model")
        .and_then(|v| v.as_str())
        .or_else(|| extra_metadata.get("ollama.model").and_then(|v| v.as_str()))
        .or_else(|| header.metadata.get("general.name").and_then(|v| v.as_str()))
        .or_else(|| path.file_stem().and_then(|s| s.to_str()))
        .unwrap_or("gguf-model");
    let architecture = header
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("gguf");

    let mut builder = AxonBuilder::new()
        .model(model)
        .architecture(architecture)
        .metadata("source_format", serde_json::json!("gguf"))
        .metadata("gguf.version", serde_json::json!(header.version))
        .metadata("gguf.tensor_count", serde_json::json!(header.tensor_count))
        .metadata("gguf.metadata", serde_json::json!(header.metadata));

    for (key, value) in extra_metadata {
        builder = builder.metadata(&key, value);
    }

    for entry in &header.tensors {
        let start = entry.data_offset as usize;
        let end = start + entry.data_size as usize;
        if end > data.len() {
            return Err(AxonError::UnexpectedEof {
                needed: end as u64,
                available: data.len() as u64,
            });
        }
        builder = builder
            .metadata(
                &format!("gguf.tensor_type.{}", entry.name),
                serde_json::json!(entry.ggml_type),
            )
            .add_tensor_unchecked(
                &entry.name,
                data[start..end].to_vec(),
                entry.dtype,
                &entry.shape,
            );
    }

    builder.build()
}

/// Resolve an Ollama model reference to the local model GGUF blob.
///
/// This reads Ollama's on-disk manifest store directly and does not require the
/// Ollama server to be running. Model references follow Ollama's common forms,
/// such as `gemma3:1b`, `qwen2.5:3b`, or `library/gemma3:1b`.
pub fn resolve_ollama_model(
    model_ref: &str,
    models_dir: Option<&Path>,
) -> AxonResult<OllamaModelSource> {
    let models_dir = models_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_ollama_models_dir);
    let (registry, namespace, model, tag) = parse_ollama_ref(model_ref);
    let manifest_path = models_dir
        .join("manifests")
        .join(registry)
        .join(namespace)
        .join(model)
        .join(tag);

    let manifest_bytes = fs::read(&manifest_path).map_err(|e| {
        AxonError::InvalidManifest(format!(
            "failed to read Ollama manifest {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| AxonError::InvalidManifest(format!("Ollama manifest JSON: {e}")))?;

    let model_layer = manifest
        .get("layers")
        .and_then(|v| v.as_array())
        .and_then(|layers| {
            layers.iter().find(|layer| {
                layer
                    .get("mediaType")
                    .and_then(|v| v.as_str())
                    .is_some_and(|media_type| media_type == "application/vnd.ollama.image.model")
            })
        })
        .ok_or_else(|| {
            AxonError::InvalidManifest(format!(
                "Ollama manifest {} does not contain a local model layer",
                manifest_path.display()
            ))
        })?;
    let digest = model_layer
        .get("digest")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AxonError::InvalidManifest("Ollama model layer missing digest".into()))?
        .to_string();
    let digest_hash = digest.strip_prefix("sha256:").unwrap_or(&digest);
    let blob_path = models_dir
        .join("blobs")
        .join(format!("sha256-{digest_hash}"));
    if !blob_path.exists() {
        return Err(AxonError::InvalidManifest(format!(
            "Ollama model blob is missing: {}",
            blob_path.display()
        )));
    }

    Ok(OllamaModelSource {
        manifest_path,
        blob_path,
        model_ref: model_ref.to_string(),
        digest,
    })
}

/// Import an Ollama-managed GGUF model into Axon format.
pub fn ollama_model_to_axon(model_ref: &str, models_dir: Option<&Path>) -> AxonResult<Vec<u8>> {
    let source = resolve_ollama_model(model_ref, models_dir)?;
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), serde_json::json!("ollama"));
    metadata.insert(
        "ollama.model".to_string(),
        serde_json::json!(source.model_ref),
    );
    metadata.insert(
        "ollama.digest".to_string(),
        serde_json::json!(source.digest),
    );
    metadata.insert(
        "ollama.manifest_path".to_string(),
        serde_json::json!(source.manifest_path.display().to_string()),
    );
    gguf_to_axon_with_metadata(&source.blob_path, metadata)
}

fn default_ollama_models_dir() -> PathBuf {
    if let Some(path) = env::var_os("OLLAMA_MODELS") {
        return PathBuf::from(path);
    }
    if let Some(path) = env::var_os("USERPROFILE") {
        return PathBuf::from(path).join(".ollama").join("models");
    }
    if let Some(path) = env::var_os("HOME") {
        return PathBuf::from(path).join(".ollama").join("models");
    }
    PathBuf::from(".ollama").join("models")
}

fn parse_ollama_ref(model_ref: &str) -> (&str, &str, &str, &str) {
    let (name_part, tag) = model_ref.rsplit_once(':').unwrap_or((model_ref, "latest"));
    let parts = name_part.split('/').collect::<Vec<_>>();
    match parts.as_slice() {
        [model] => ("registry.ollama.ai", "library", *model, tag),
        [namespace, model] => ("registry.ollama.ai", *namespace, *model, tag),
        [registry, namespace, model] => (*registry, *namespace, *model, tag),
        _ => ("registry.ollama.ai", "library", name_part, tag),
    }
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> AxonResult<u32> {
    use byteorder::ReadBytesExt;
    Ok(cursor.read_u32::<byteorder::LittleEndian>()?)
}

fn read_u64(cursor: &mut Cursor<&[u8]>) -> AxonResult<u64> {
    use byteorder::ReadBytesExt;
    Ok(cursor.read_u64::<byteorder::LittleEndian>()?)
}

fn read_i8(cursor: &mut Cursor<&[u8]>) -> AxonResult<i8> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
}

fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> AxonResult<String> {
    let len = read_u64(cursor)? as usize;
    let start = cursor.position() as usize;
    let end = start
        .checked_add(len)
        .ok_or_else(|| AxonError::InvalidManifest("GGUF string length overflow".to_string()))?;
    if end > cursor.get_ref().len() {
        return Err(AxonError::UnexpectedEof {
            needed: end as u64,
            available: cursor.get_ref().len() as u64,
        });
    }
    let value = String::from_utf8_lossy(&cursor.get_ref()[start..end]).into_owned();
    cursor.set_position(end as u64);
    Ok(value)
}

fn read_gguf_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> AxonResult<serde_json::Value> {
    use byteorder::ReadBytesExt;
    Ok(match value_type {
        GGUF_TYPE_U8 => serde_json::json!(cursor.read_u8()?),
        GGUF_TYPE_I8 => serde_json::json!(read_i8(cursor)?),
        GGUF_TYPE_U16 => serde_json::json!(cursor.read_u16::<byteorder::LittleEndian>()?),
        GGUF_TYPE_I16 => serde_json::json!(cursor.read_i16::<byteorder::LittleEndian>()?),
        GGUF_TYPE_U32 => serde_json::json!(cursor.read_u32::<byteorder::LittleEndian>()?),
        GGUF_TYPE_I32 => serde_json::json!(cursor.read_i32::<byteorder::LittleEndian>()?),
        GGUF_TYPE_F32 => serde_json::json!(cursor.read_f32::<byteorder::LittleEndian>()?),
        GGUF_TYPE_BOOL => serde_json::json!(cursor.read_u8()? != 0),
        GGUF_TYPE_STRING => serde_json::json!(read_gguf_string(cursor)?),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(cursor)?;
            let len = read_u64(cursor)?;
            let mut values = Vec::with_capacity(len.min(1024) as usize);
            for _ in 0..len {
                values.push(read_gguf_value(cursor, elem_type)?);
            }
            serde_json::Value::Array(values)
        }
        GGUF_TYPE_U64 => serde_json::json!(cursor.read_u64::<byteorder::LittleEndian>()?),
        GGUF_TYPE_I64 => serde_json::json!(cursor.read_i64::<byteorder::LittleEndian>()?),
        GGUF_TYPE_F64 => serde_json::json!(cursor.read_f64::<byteorder::LittleEndian>()?),
        other => {
            return Err(AxonError::InvalidManifest(format!(
                "unsupported GGUF metadata value type {other}"
            )));
        }
    })
}

fn ggml_type_to_dtype(ggml_type: u32) -> DType {
    match ggml_type {
        0 => DType::F32,
        1 => DType::F16,
        2 | 3 | 12 | 13 | 14 | 15 | 16 | 17 | 18 => DType::Q4,
        8 | 10 | 11 | 19 | 20 | 21 | 22 => DType::Q8,
        24 => DType::I8,
        25 => DType::I16,
        26 => DType::I32,
        _ => DType::U8,
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

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
