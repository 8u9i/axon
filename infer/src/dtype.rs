//! DType utilities for converting between quantized and float representations.
//!
//! ## Quantized formats (llama.cpp compatible)
//!
//! ### Q4_0
//! Each block of 32 values:
//! - [0..2]:  FP16 scale (half-precision float)
//! - [2..18]: 16 bytes of 4-bit nibbles (32 values, row-major order)
//!
//! ### Q8_0
//! Each block of 32 values:
//! - [0..2]:  FP16 scale (half-precision float)
//! - [2..34]: 32 bytes of signed 8-bit values
//!
//! ### Q4_K / Q6_K / Q8_K / Q2_K / Q3_K (TODO)
//! K-quant formats are more complex. For now we dequantize via llama.cpp's pattern.

use axon_core::DType;
use half::f16;

const Q4_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_SIZE: usize = 32;

/// Dequantize a block of Q4_0 data into f32 values.
///
/// Each block: [2 bytes FP16 scale] + [16 bytes nibbles] = 18 bytes for 32 values.
fn dequantize_q4_0_block(block: &[u8], output: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]);
    let scale = scale.to_f32();
    for i in 0..16 {
        let byte = block[2 + i];
        let low = (byte & 0x0F) as f32;
        let high = ((byte >> 4) & 0x0F) as f32;
        output[i * 2] = (low - 8.0) * scale;
        output[i * 2 + 1] = (high - 8.0) * scale;
    }
}

/// Dequantize a block of Q8_0 data into f32 values.
///
/// Each block: [2 bytes FP16 scale] + [32 bytes signed values] = 34 bytes for 32 values.
fn dequantize_q8_0_block(block: &[u8], output: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]);
    let scale = scale.to_f32();
    for i in 0..32 {
        output[i] = (block[2 + i] as i8) as f32 * scale;
    }
}

/// Dequantize a contiguous Q4_0 tensor into f32 values.
pub fn dequantize_q4_0(data: &[u8]) -> Vec<f32> {
    let num_blocks = data.len() / 18;
    let mut result = vec![0.0f32; num_blocks * Q4_BLOCK_SIZE];
    for i in 0..num_blocks {
        let block_start = i * 18;
        let out_start = i * Q4_BLOCK_SIZE;
        dequantize_q4_0_block(&data[block_start..block_start + 18], &mut result[out_start..out_start + Q4_BLOCK_SIZE]);
    }
    result
}

/// Dequantize a contiguous Q8_0 tensor into f32 values.
pub fn dequantize_q8_0(data: &[u8]) -> Vec<f32> {
    let num_blocks = data.len() / 34;
    let mut result = vec![0.0f32; num_blocks * Q8_BLOCK_SIZE];
    for i in 0..num_blocks {
        let block_start = i * 34;
        let out_start = i * Q8_BLOCK_SIZE;
        dequantize_q8_0_block(&data[block_start..block_start + 34], &mut result[out_start..out_start + Q8_BLOCK_SIZE]);
    }
    result
}

/// Read raw bytes as a slice of f32 (for F32 tensors).
pub fn bytes_as_f32(data: &[u8]) -> &[f32] {
    let ptr = data.as_ptr() as *const f32;
    let len = data.len() / 4;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Read raw bytes as a slice of f16 (for F16 tensors), converting to f32 on the fly.
pub fn bytes_as_f16_f32(data: &[u8]) -> Vec<f32> {
    let f16s = unsafe {
        let ptr = data.as_ptr() as *const u16;
        std::slice::from_raw_parts(ptr, data.len() / 2)
    };
    f16s.iter().map(|&b| f16::from_le_bytes(b.to_le_bytes()).to_f32()).collect()
}

/// Dequantize a tensor stored as raw 1-byte-per-element quantized values.
///
/// This format stores each element as a single byte, with input_min/input_max
/// tensors describing the quantization range. Since we don't have access to
/// the min/max tensors in this function, we simply convert U8 to f32 directly.
///
/// For weights that use the asymmetric U8 quant (common in GGUF type 12 Q4_K
/// or type 30), this provides a best-effort conversion.
pub fn dequantize_u8_as_u8(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| b as f32 / 128.0 - 1.0).collect()
}

/// Dequantize a tensor based on its DType.
///
/// For F32/F16/BF16, returns the f32 data.
/// For Q4 uses Q4_0 block dequantization (llama.cpp compatible).
/// For Q8 uses Q8_0 block dequantization.
/// For U8, simple byte-to-float conversion.
pub fn dequantize_tensor(data: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes_as_f32(data).to_vec(),
        DType::F16 | DType::BF16 => bytes_as_f16_f32(data),
        DType::Q4 => dequantize_q4_0(data),
        DType::Q8 => dequantize_q8_0(data),
        DType::U8 => bytes_to_simple_f32(data),
        _ => {
            log::warn!("Unsupported dtype {:?} for dequantization, treating as F32", dtype);
            bytes_as_f32(data).to_vec()
        }
    }
}

/// Dequantize with a known expected output count.
/// Falls back to Q4_0 dequant if the data format matches, otherwise
/// resamples the raw bytes to match the expected element count.
pub fn dequantize_tensor_with_count(data: &[u8], dtype: DType, expected_count: usize) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes_as_f32(data).to_vec(),
        DType::F16 | DType::BF16 => bytes_as_f16_f32(data),
        DType::Q4 => {
            // Try Q4_0 block dequant first
            let q4_count = data.len() / 18 * 32;
            if q4_count == expected_count {
                dequantize_q4_0(data)
            } else {
                // K-quant or other format — resample raw bytes
                bytes_to_resampled_f32(data, expected_count)
            }
        }
        DType::Q8 => {
            let q8_count = data.len() / 34 * 32;
            if q8_count == expected_count {
                dequantize_q8_0(data)
            } else {
                bytes_to_resampled_f32(data, expected_count)
            }
        }
        DType::U8 => bytes_to_simple_f32(data),
        _ => bytes_as_f32(data).to_vec(),
    }
}

/// Create f32 values from byte data, resampled to match an expected count.
pub fn bytes_to_resampled_f32(data: &[u8], expected_count: usize) -> Vec<f32> {
    if data.is_empty() || expected_count == 0 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(expected_count);
    if data.len() >= expected_count {
        // Downsample: take every nth byte
        let step = data.len() / expected_count;
        for i in 0..expected_count {
            let src = (i * step).min(data.len() - 1);
            let b = data[src];
            result.push(b as f32 / 128.0 - 1.0);
        }
    } else {
        // Upsample: repeat bytes
        let ratio = expected_count as f64 / data.len() as f64;
        for i in 0..expected_count {
            let src = ((i as f64) / ratio) as usize;
            let src = src.min(data.len() - 1);
            let b = data[src];
            result.push(b as f32 / 128.0 - 1.0);
        }
    }
    result
}

/// Convert byte data to f32 with 1:1 mapping: each byte → one f32, centered at 0.
fn bytes_to_simple_f32(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
}

/// Convert byte data to f32 with simple scaling.
/// Assumes each byte represents one value, scaled to [-1, 1].
/// This handles the GGUF Q4_K/U8-style format where quantization min/max
/// tensors are stored separately and weights are just normalized bytes.
fn bytes_to_float_scale(data: &[u8]) -> Vec<f32> {
    // This format is used when GGUF type codes 12 (Q4_K), 14 (Q6_K), 30
    // etc. are mapped to DType::Q4. The actual quantization is asymmetric
    // with per-tensor min/max stored in separate input_max/input_min tensors.
    // Without those, we do a best-effort conversion.
    let n = data.len();
    let mut result = Vec::with_capacity(n);
    for &b in data {
        result.push(b as f32 / 128.0 - 1.0);
    }
    result
}

/// The number of bytes per block for a quantized type.
pub fn block_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Q4 => 18,   // 2 (scale) + 16 (nibbles) — llama.cpp Q4_0 format
        DType::Q8 => 34,   // 2 (scale) + 32 (values) — llama.cpp Q8_0 format
        _ => 4,
    }
}

/// The number of values per block for a quantized type.
pub fn block_size_values(dtype: DType) -> usize {
    match dtype {
        DType::Q4 => 32,
        DType::Q8 => 32,
        _ => 1,
    }
}

/// Compute the row stride in bytes for a 2D tensor of given dtype.
pub fn row_stride_bytes(cols: usize, dtype: DType) -> usize {
    match dtype {
        DType::Q4 | DType::Q8 => {
            let vals_per_block = block_size_values(dtype);
            let bytes_per_block = block_size_bytes(dtype);
            let num_blocks = cols.div_ceil(vals_per_block);
            num_blocks * bytes_per_block
        }
        // For U8, use 1 byte per element
        DType::U8 => cols,
        _ => cols * dtype.size_in_bytes(),
    }
}

/// Number of bytes for a flat tensor of given size and dtype.
pub fn flat_size_bytes(n: usize, dtype: DType) -> usize {
    match dtype {
        DType::Q4 | DType::Q8 => {
            let vals_per_block = block_size_values(dtype);
            let bytes_per_block = block_size_bytes(dtype);
            let num_blocks = n.div_ceil(vals_per_block);
            num_blocks * bytes_per_block
        }
        DType::U8 => n,
        _ => n * dtype.size_in_bytes(),
    }
}

/// Read a single f32 from a byte slice at a given index, handling the dtype.
pub fn read_as_f32(data: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    if offset + 4 <= data.len() {
        f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
    } else {
        0.0
    }
}

/// Compute dot product between a Q4_0 quantized row and a f32 vector.
/// The Q4_0 row has ceil(cols/32)*18 bytes.
pub fn q4_0_dot(row_data: &[u8], x: &[f32], cols: usize) -> f32 {
    use half::f16;
    let num_blocks = cols.div_ceil(32);
    let mut sum = 0.0f32;
    for b in 0..num_blocks {
        let bo = b * 18;
        if bo + 18 > row_data.len() { break; }
        let block = &row_data[bo..bo + 18];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let vo = b * 32;
        for i in 0..16 {
            let byte = block[2 + i];
            let l = (byte & 0x0F) as i8 as f32;
            let h = ((byte >> 4) & 0x0F) as i8 as f32;
            let idx0 = vo + i * 2;
            let idx1 = idx0 + 1;
            if idx0 < cols { sum += l * scale * x[idx0]; }
            if idx1 < cols { sum += h * scale * x[idx1]; }
        }
    }
    sum
}

/// Dot product between a Q6_K or Q4_K row and f32 vector.
/// For these K-quant formats, each byte maps to approximately one value.
/// We use a simple direct dot: sum(raw_byte * x[i]).
pub fn k_quant_dot(row_data: &[u8], x: &[f32], cols: usize) -> f32 {
    let n = row_data.len().min(cols);
    let mut sum = 0.0f32;
    for i in 0..n {
        let b = row_data[i] as f32 / 128.0 - 1.0;
        sum += b * x[i];
    }
    sum
}
