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

/// Dequantize a tensor based on its DType.
///
/// For F32/F16/BF16, returns the f32 data.
/// For Q4/Q8, dequantizes to f32.
pub fn dequantize_tensor(data: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytes_as_f32(data).to_vec(),
        DType::F16 | DType::BF16 => bytes_as_f16_f32(data),
        DType::Q4 => dequantize_q4_0(data),
        DType::Q8 => dequantize_q8_0(data),
        _ => {
            log::warn!("Unsupported dtype {:?} for dequantization, treating as F32", dtype);
            bytes_as_f32(data).to_vec()
        }
    }
}

/// The number of bytes per block for a quantized type.
pub fn block_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Q4 => 18,   // 2 (scale) + 16 (nibbles)
        DType::Q8 => 34,   // 2 (scale) + 32 (values)
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
