//! Quantized matrix operations.
//!
//! Implements Q4_0 and Q8_0 matrix-vector multiply directly on packed data,
//! without dequantizing the entire weight matrix.

use half::f16;

const Q4_BLOCK_SIZE: usize = 32;
const Q4_BLOCK_BYTES: usize = 18; // 2 (scale) + 16 (nibbles)

const Q8_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_BYTES: usize = 34; // 2 (scale) + 32 (values)

// ── Q4_0 helpers ───────────────────────────────────────────────────

/// Dequantize a single Q4_0 block to f32, writing into output[0..32].
#[inline]
fn deq_q4_block(block: &[u8], output: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scale = scale;
    for i in 0..16 {
        let byte = block[2 + i];
        let l = (byte & 0x0F) as i8 as f32;
        let h = ((byte >> 4) & 0x0F) as i8 as f32;
        output[i * 2] = l * scale;
        output[i * 2 + 1] = h * scale;
    }
}

/// Dot product between a Q4_0 weight row and a dequantized f32 input vector.
///
/// Row data: sequence of Q4_0 blocks (18 bytes each, 32 values per block)
/// Input: f32 values (floats)
/// Output: single f32 dot product result
pub fn q4_dot(row_data: &[u8], input: &[f32], cols: usize) -> f32 {
    let num_blocks = cols.div_ceil(Q4_BLOCK_SIZE);
    let mut deq_buf = [0.0f32; Q4_BLOCK_SIZE];
    let mut sum = 0.0f32;

    for b in 0..num_blocks {
        let block_start = b * Q4_BLOCK_BYTES;
        if block_start + Q4_BLOCK_BYTES > row_data.len() {
            break;
        }
        deq_q4_block(&row_data[block_start..], &mut deq_buf);
        let val_start = b * Q4_BLOCK_SIZE;
        for i in 0..Q4_BLOCK_SIZE {
            let idx = val_start + i;
            if idx < cols {
                sum += deq_buf[i] * input[idx];
            }
        }
    }
    sum
}

/// Q4_0 matrix-vector multiply: y = A @ x
///
/// - A: [rows, cols] stored as Q4_0 blocks, row-major
///   Each row: ceil(cols/32) * 18 bytes
/// - x: [cols] f32 input
/// - y: [rows] f32 output
pub fn q4_matvec(row_data: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let row_stride_bytes = cols.div_ceil(Q4_BLOCK_SIZE) * Q4_BLOCK_BYTES;
    for r in 0..rows {
        let row_start = r * row_stride_bytes;
        let row_end = (row_start + row_stride_bytes).min(row_data.len());
        y[r] = q4_dot(&row_data[row_start..row_end], x, cols);
    }
}

// ── Q8_0 helpers ───────────────────────────────────────────────────

/// Dequantize a single Q8_0 block to f32, writing into output[0..32].
#[inline]
fn deq_q8_block(block: &[u8], output: &mut [f32]) {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..32 {
        output[i] = (block[2 + i] as i8) as f32 * scale;
    }
}

/// Q8_0 matrix-vector multiply: y = A @ x
pub fn q8_matvec(row_data: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let row_stride_bytes = cols.div_ceil(Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES;
    let mut deq_buf = [0.0f32; Q8_BLOCK_SIZE];
    let num_blocks = cols.div_ceil(Q8_BLOCK_SIZE);

    for r in 0..rows {
        let row_start = r * row_stride_bytes;
        let row_end = (row_start + row_stride_bytes).min(row_data.len());
        let row = &row_data[row_start..row_end];
        let mut sum = 0.0f32;
        for b in 0..num_blocks {
            let block_start = b * Q8_BLOCK_BYTES;
            if block_start + Q8_BLOCK_BYTES > row.len() {
                break;
            }
            deq_q8_block(&row[block_start..], &mut deq_buf);
            let val_start = b * Q8_BLOCK_SIZE;
            for i in 0..Q8_BLOCK_SIZE {
                let idx = val_start + i;
                if idx < cols {
                    sum += deq_buf[i] * x[idx];
                }
            }
        }
        y[r] = sum;
    }
}

// ── Generic quantized matvec ───────────────────────────────────────

/// Dispatch to the correct quantized matvec based on the block type.
///
/// Returns the number of bytes per row in the quantized format.
pub fn quantized_matvec(
    row_data: &[u8],
    rows: usize,
    cols: usize,
    block_bytes: usize,
    vals_per_block: usize,
    x: &[f32],
    y: &mut [f32],
) {
    let row_stride = cols.div_ceil(vals_per_block) * block_bytes;
    let mut deq_buf = vec![0.0f32; vals_per_block];
    let num_blocks = cols.div_ceil(vals_per_block);

    for r in 0..rows {
        let row_start = r * row_stride;
        let row_end = (row_start + row_stride).min(row_data.len());
        let row = &row_data[row_start..row_end];
        let mut sum = 0.0f32;

        for b in 0..num_blocks {
            let block_start = b * block_bytes;
            if block_start + block_bytes > row.len() {
                break;
            }
            let block = &row[block_start..block_start + block_bytes];

            // Handle each quantized type
            if block_bytes == Q4_BLOCK_BYTES && vals_per_block == Q4_BLOCK_SIZE {
                // Q4_0
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                for i in 0..16 {
                    let byte = block[2 + i];
                    let l = (byte & 0x0F) as i8 as f32;
                    let h = ((byte >> 4) & 0x0F) as i8 as f32;
                    deq_buf[i * 2] = l * scale;
                    deq_buf[i * 2 + 1] = h * scale;
                }
            } else if block_bytes == Q8_BLOCK_BYTES && vals_per_block == Q8_BLOCK_SIZE {
                // Q8_0
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                for i in 0..32 {
                    deq_buf[i] = (block[2 + i] as i8) as f32 * scale;
                }
            } else {
                log::warn!("Unknown quantized block format: {}/{}", block_bytes, vals_per_block);
                continue;
            }

            let val_start = b * vals_per_block;
            for i in 0..vals_per_block {
                let idx = val_start + i;
                if idx < cols {
                    sum += deq_buf[i] * x[idx];
                }
            }
        }
        y[r] = sum;
    }
}
