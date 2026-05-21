//! # Tensor Slicing and Partial Loading
//!
//! Provides shape-aware slicing for tensors stored in `.axon` files.
//! The key design goal: **load only the bytes you need** from the mmap,
//! avoiding pulling the entire tensor into memory.
//!
//! ## Supported slice types
//!
//! - **Byte range** — raw offset + size in bytes
//! - **Row range** — load contiguous rows from a 2D row-major tensor
//! - **Row + column range** — load a rectangular submatrix
//! - **Offset-based** — element-index-based slicing for arbitrary dimensions
//!
//! ## Memory semantics
//!
//! Slicing reads only the requested bytes from the mmap. The OS faults in
//! the corresponding file pages. Unused portions of the tensor remain on
//! disk (or in the OS page cache, depending on access patterns).
//!
//! ## Row-major assumption
//!
//! ML frameworks store weight matrices in row-major order. The slicing
//! implementation assumes row-major layout. Column-major support can be
//! added later via a `layout` parameter.

use axon_core::AxonError;
use axon_core::{DType, AxonResult};

/// Specifies which portion of a tensor to load.
///
/// The spec is validated against the tensor's shape and dtype before
/// any bytes are read from the mmap.
#[derive(Debug, Clone)]
pub enum SliceSpec {
    /// Load a contiguous byte range: `[byte_offset, byte_offset + size)`.
    Bytes {
        byte_offset: u64,
        size: u64,
    },
    /// Load contiguous rows from a 2D tensor.
    /// `row_start` is inclusive, `row_end` is exclusive.
    Rows {
        row_start: u64,
        row_end: u64,
    },
    /// Load a rectangular submatrix from a 2D tensor.
    /// All ranges are `[start, end)`.
    RowCol {
        row_start: u64,
        row_end: u64,
        col_start: u64,
        col_end: u64,
    },
    /// Load elements by offset and count.
    /// `element_offset` is the index of the first element (not byte).
    Elements {
        element_offset: u64,
        count: u64,
    },
}

impl SliceSpec {
    /// Create a byte-range slice.
    pub fn byte_range(byte_offset: u64, size: u64) -> Self {
        SliceSpec::Bytes { byte_offset, size }
    }

    /// Create a row-range slice for a 2D tensor.
    /// `start` is inclusive, `end` is exclusive.
    pub fn rows(start: u64, end: u64) -> Self {
        SliceSpec::Rows { row_start: start, row_end: end }
    }

    /// Create a rectangular submatrix slice.
    pub fn row_col(row_start: u64, row_end: u64, col_start: u64, col_end: u64) -> Self {
        SliceSpec::RowCol { row_start, row_end, col_start, col_end }
    }

    /// Validate this spec against a tensor descriptor and compute the byte range.
    ///
    /// Returns `(byte_offset, byte_size)` if valid.
    pub fn resolve(&self, dtype: DType, shape: &[u64], data_offset: u64, data_size: u64) -> AxonResult<(u64, u64)> {
        let elem_size = dtype.size_in_bytes() as u64;

        match self {
            SliceSpec::Bytes { byte_offset, size } => {
                if byte_offset + size > data_size {
                    return Err(AxonError::UnexpectedEof {
                        needed: data_offset + byte_offset + size,
                        available: data_offset + data_size,
                    });
                }
                Ok((data_offset + byte_offset, *size))
            }

            SliceSpec::Rows { row_start, row_end } => {
                if shape.len() != 2 {
                    return Err(AxonError::InvalidManifest(
                        format!("Rows slice requires 2D tensor, got {}D", shape.len())
                    ));
                }
                let cols = shape[1];
                let row_stride = cols * elem_size;
                if *row_end > shape[0] || *row_start > *row_end {
                    return Err(AxonError::UnexpectedEof {
                        needed: data_offset + row_stride * shape[0],
                        available: data_offset + data_size,
                    });
                }
                let byte_offset = row_start * row_stride;
                let size = (row_end - row_start) * row_stride;
                Ok((data_offset + byte_offset, size))
            }

            SliceSpec::RowCol { row_start, row_end, col_start, col_end } => {
                if shape.len() != 2 {
                    return Err(AxonError::InvalidManifest(
                        format!("RowCol slice requires 2D tensor, got {}D", shape.len())
                    ));
                }
                let cols = shape[1];
                let row_stride = cols * elem_size;
                if *row_end > shape[0] || *col_end > cols || *row_start > *row_end || *col_start > *col_end {
                    return Err(AxonError::UnexpectedEof {
                        needed: data_offset + row_stride * shape[0] + col_end * elem_size,
                        available: data_offset + data_size,
                    });
                }
                let byte_offset = row_start * row_stride + col_start * elem_size;
                let width = (col_end - col_start) * elem_size;
                let height = row_end - row_start;
                let size = width * height;
                Ok((data_offset + byte_offset, size))
            }

            SliceSpec::Elements { element_offset, count } => {
                let total_elements: u64 = shape.iter().product();
                if element_offset + count > total_elements {
                    return Err(AxonError::UnexpectedEof {
                        needed: data_offset + (element_offset + count) * elem_size,
                        available: data_offset + data_size,
                    });
                }
                let byte_offset = element_offset * elem_size;
                let size = count * elem_size;
                Ok((data_offset + byte_offset, size))
            }
        }
    }
}

/// Shape-aware tensor slice result.
/// Contains both the raw bytes and the logical shape of the slice.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    /// Raw bytes from the mmap.
    pub data: Vec<u8>,
    /// The logical shape of this slice (e.g., `[10, 4096]` for 10 rows).
    pub shape: Vec<u64>,
    /// The dtype of the elements.
    pub dtype: DType,
}

impl TensorSlice {
    pub fn new(data: Vec<u8>, shape: Vec<u64>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    /// Number of elements in this slice.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{AxonBuilder, TensorDescriptor, Affinity};

    fn make_2d_tensor(rows: u64, cols: u64) -> (Vec<u8>, Vec<u8>) {
        let dtype = DType::F32;
        let elem_size = dtype.size_in_bytes() as u64;
        let total = (rows * cols * elem_size) as usize;
        let data: Vec<u8> = (0..total).map(|i| i as u8).collect();
        let _desc = TensorDescriptor::new(
            "weights", dtype, &[rows, cols],
            4096, total as u64, Affinity::Default, 0,
        );

        // Build a minimal .axon file and extract the raw data bytes
        let axon = AxonBuilder::new()
            .add_tensor("weights", data.clone(), dtype, &[rows, cols])
            .build()
            .unwrap();

        // Parse it back to get correct offsets
        let file = axon_core::AxonFile::from_bytes(axon).unwrap();
        let raw = file.tensor_data("weights").unwrap().to_vec();

        (raw.clone(), raw)
    }

    #[test]
    fn test_byte_range_resolve() {
        let spec = SliceSpec::byte_range(16, 32);
        let (offset, size) = spec.resolve(DType::F32, &[64, 64], 4096, 16384).unwrap();
        assert_eq!(offset, 4112);  // 4096 + 16
        assert_eq!(size, 32);
    }

    #[test]
    fn test_rows_resolve() {
        // 64 rows × 32 cols × 4 bytes = 8192 bytes
        let spec = SliceSpec::rows(10, 20);
        let elem_size = DType::F32.size_in_bytes() as u64;
        let row_stride = 32 * elem_size;
        let (offset, size) = spec.resolve(DType::F32, &[64, 32], 4096, 8192).unwrap();
        assert_eq!(offset, 4096 + 10 * row_stride);
        assert_eq!(size, 10 * row_stride);
    }

    #[test]
    fn test_rowcol_resolve() {
        let spec = SliceSpec::row_col(5, 10, 8, 16);
        let elem_size = DType::F32.size_in_bytes() as u64;
        let row_stride = 32 * elem_size;
        let (offset, size) = spec.resolve(DType::F32, &[64, 32], 4096, 8192).unwrap();
        assert_eq!(offset, 4096 + 5 * row_stride + 8 * elem_size);
        // 5 rows × 8 columns × 4 bytes
        assert_eq!(size, 5 * 8 * elem_size);
    }

    #[test]
    fn test_elements_resolve() {
        let spec = SliceSpec::Elements { element_offset: 100, count: 50 };
        let elem_size = DType::F32.size_in_bytes() as u64;
        let (offset, size) = spec.resolve(DType::F32, &[4096], 4096, 16384).unwrap();
        assert_eq!(offset, 4096 + 100 * elem_size);
        assert_eq!(size, 50 * elem_size);
    }

    #[test]
    fn test_rows_on_1d_tensor_fails() {
        let spec = SliceSpec::rows(0, 10);
        let result = spec.resolve(DType::F32, &[4096], 4096, 16384);
        assert!(result.is_err());
    }

    #[test]
    fn test_byte_range_beyond_size_fails() {
        let spec = SliceSpec::byte_range(0, 999999);
        let result = spec.resolve(DType::F32, &[64, 64], 4096, 16384);
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_slice_matches_full_values() {
        // Build an AxonRuntime and compare slice values against full load
        use std::fs;
        use std::path::PathBuf;
        use crate::AxonRuntime;

        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let dir = PathBuf::from("output");
        fs::create_dir_all(&dir).ok();
        let path = dir.join("slice_2d_test.axon");
        let axon = AxonBuilder::new()
            .add_tensor("mat", data.clone(), DType::U8, &[16, 16])
            .build()
            .unwrap();
        fs::write(&path, &axon).unwrap();

        let rt = AxonRuntime::open(&path).unwrap();

        // Full load
        let full = rt.tensor("mat").unwrap();

        // Slice: rows 4..8. Resolve against the tensor's content bounds (0, 256)
        // to get (byte_offset_within_tensor, slice_size).
        let spec = SliceSpec::rows(4, 8);
        let (byte_offset_within_tensor, sz) = spec.resolve(DType::U8, &[16, 16], 0, 256).unwrap();
        let sliced = rt.tensor_byte_range("mat", byte_offset_within_tensor, sz).unwrap();
        assert_eq!(sliced.len(), 4 * 16);  // 4 rows × 16 cols
        assert_eq!(&sliced, &full[4*16..8*16]);

        // Clean up
        fs::remove_file(&path).ok();
    }
}
