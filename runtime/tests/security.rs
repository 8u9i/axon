//! Security and corruption tests for the Axon Runtime.
//!
//! Malformed .axon files must never panic or crash the process.
//! Every error must be returned as a structured `AxonError`.

use std::fs;
use std::path::PathBuf;

use axon_core::{AxonBuilder, DType, AxonFile, AxonHeader, TensorDescriptor};
use axon_runtime::AxonRuntime;

fn test_dir() -> PathBuf {
    let dir = PathBuf::from("output").join("security");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Write raw bytes as an .axon file.
fn write_raw(path: &PathBuf, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
}

/// Build a minimal valid .axon file and return its bytes.
fn build_valid_axon() -> Vec<u8> {
    let data = vec![1u8; 64];
    AxonBuilder::new()
        .model("security-test")
        .architecture("test")
        .add_tensor("weights", data, DType::U8, &[64])
        .build()
        .unwrap()
}

#[test]
fn test_corrupt_header_magic() {
    let path = test_dir().join("corrupt_magic.axon");
    let mut bytes = build_valid_axon();
    bytes[0] = 0xFF; // Corrupt first magic byte
    bytes[1] = 0xFE;
    bytes[2] = 0xFD;
    bytes[3] = 0xFC;
    write_raw(&path, &bytes);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Corrupt magic should fail");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("magic"), "Error should mention magic: {msg}");
}

#[test]
fn test_corrupt_header_version() {
    let path = test_dir().join("corrupt_version.axon");
    let mut bytes = build_valid_axon();
    // Version is u32 at offset 4
    bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    write_raw(&path, &bytes);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Unsupported version should fail");
}

#[test]
fn test_truncated_file() {
    let path = test_dir().join("truncated.axon");
    let full = build_valid_axon();
    // Only first 40 bytes — not enough for a valid header
    write_raw(&path, &full[..40]);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Truncated header should fail");
}

#[test]
fn test_truncated_after_header() {
    let path = test_dir().join("truncated_after_header.axon");
    let full = build_valid_axon();
    // Only 64 bytes — header only, no manifest
    write_raw(&path, &full[..64]);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Header-only file should fail");
}

#[test]
fn test_empty_file() {
    let path = test_dir().join("empty.axon");
    write_raw(&path, &[]);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Empty file should fail");
}

#[test]
fn test_zero_tensor_count_with_payload() {
    let path = test_dir().join("zero_tensors.axon");
    let mut bytes = build_valid_axon();
    // Set tensor_count to 0 at offset 24
    bytes[24..32].copy_from_slice(&0u64.to_le_bytes());
    write_raw(&path, &bytes);

    // Should either parse (empty model) or fail gracefully
    let result = AxonRuntime::open(&path);
    // No panic — either works or returns error
    if let Err(e) = result {
        let msg = e.to_string();
        assert!(!msg.is_empty(), "Error message should not be empty");
    }
}

#[test]
fn test_huge_tensor_count() {
    let path = test_dir().join("huge_count.axon");
    let mut bytes = build_valid_axon();
    // Set tensor_count to a huge value
    bytes[24..32].copy_from_slice(&(1_000_000u64).to_le_bytes());
    write_raw(&path, &bytes);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Huge tensor count should fail");
}

#[test]
fn test_invalid_tensor_name_lookup() {
    let path = test_dir().join("invalid_name.axon");
    let bytes = build_valid_axon();
    write_raw(&path, &bytes);

    let rt = AxonRuntime::open(&path).unwrap();

    // Various invalid names — should all return TensorNotFound, not panic
    assert!(rt.tensor_view("").is_err());
    assert!(rt.tensor_view("\0").is_err());
    assert!(rt.tensor_view("nonexistent_tensor_that_doesnt_exist").is_err());
    assert!(rt.tensor_view("layer_0_weight\x00extra").is_err());
}

#[test]
fn test_negative_offset_does_not_crash() {
    // Ensure accessing a large/negative offset returns error, not panic
    let path = test_dir().join("offset_test.axon");
    let mut bytes = build_valid_axon();

    // Parse and modify first descriptor's data_offset to a very large value
    // First find where the TDT starts
    let header = AxonHeader::from_bytes(&bytes[..64]).unwrap();
    let tdt_start = (header.manifest_offset + header.manifest_size + 63) & !63;
    let desc_start = tdt_start as usize;

    if desc_start + 192 <= bytes.len() {
        // Set data_offset to a value beyond file size
        let huge_offset: u64 = 1_000_000_000_000;
        let offset_pos = desc_start + 136;  // data_offset field at byte 136
        bytes[offset_pos..offset_pos + 8].copy_from_slice(&huge_offset.to_le_bytes());
        write_raw(&path, &bytes);

        let rt = AxonRuntime::open(&path).unwrap();
        let result = rt.tensor_view("weights");
        assert!(result.is_err(), "Offset beyond file should fail");
    }
}

#[test]
fn test_interleaved_tensor_names() {
    // Verify tensor lookup is exact — partial prefix matches must fail
    let path = test_dir().join("interleaved_names.axon");
    let mut builder = AxonBuilder::new();
    builder = builder.add_tensor("abc.weight", vec![0u8; 16], DType::U8, &[16]);
    builder = builder.add_tensor("abcdef.weight", vec![1u8; 16], DType::U8, &[16]);
    let bytes = builder.build().unwrap();
    write_raw(&path, &bytes);

    let rt = AxonRuntime::open(&path).unwrap();

    // "abc" should not match "abc.weight"
    assert!(rt.tensor_view("abc").is_err(),
            "Partial name 'abc' should not match 'abc.weight'");
    assert!(rt.tensor_view("abc.weight").is_ok(),
            "Exact name 'abc.weight' should match");
}

#[test]
fn test_corrupt_manifest_json() {
    let path = test_dir().join("corrupt_manifest.axon");
    let mut bytes = build_valid_axon();

    // Corrupt the manifest area with garbage
    let header = AxonHeader::from_bytes(&bytes[..64]).unwrap();
    let manifest_start = header.manifest_offset as usize;
    let manifest_end = manifest_start + header.manifest_size as usize;
    if manifest_end <= bytes.len() {
        for i in manifest_start..manifest_end {
            bytes[i] = 0xFF;
        }
    }
    write_raw(&path, &bytes);

    let result = AxonRuntime::open(&path);
    assert!(result.is_err(), "Corrupt manifest should fail");
}

#[test]
fn test_checksum_mismatch() {
    let path = test_dir().join("checksum_mismatch.axon");
    let mut bytes = build_valid_axon();
    let header = AxonHeader::from_bytes(&bytes[..64]).unwrap();
    let manifest_start = header.manifest_offset as usize;
    if manifest_start + 10 < bytes.len() {
        bytes[manifest_start + 10] ^= 0xFF;
    }
    write_raw(&path, &bytes);
    let result = AxonRuntime::open(&path);
    // Should not crash regardless of checksum
    drop(result);
}

#[test]
fn test_shape_overflow() {
    let path = test_dir().join("shape_overflow.axon");
    let mut bytes = build_valid_axon();
    let header = AxonHeader::from_bytes(&bytes[..64]).unwrap();
    let tdt_start = (header.manifest_offset + header.manifest_size + 63) & !63;
    let desc_start = tdt_start as usize;
    if desc_start + 192 <= bytes.len() {
        bytes[desc_start + 130..desc_start + 138].copy_from_slice(&1_000_000u64.to_le_bytes());
        write_raw(&path, &bytes);
        let rt = AxonRuntime::open(&path).unwrap();
        let result = rt.tensor_view("weights");
        drop(result);
    }
}

#[test]
fn test_invalid_dtype_code() {
    let path = test_dir().join("invalid_dtype.axon");
    let mut bytes = build_valid_axon();
    let header = AxonHeader::from_bytes(&bytes[..64]).unwrap();
    let tdt_start = (header.manifest_offset + header.manifest_size + 63) & !63;
    let desc_start = tdt_start as usize;
    if desc_start + 192 <= bytes.len() {
        bytes[desc_start + 128] = 255;
        write_raw(&path, &bytes);
        let rt = AxonRuntime::open(&path).unwrap();
        let result = rt.tensor_view("weights");
        drop(result);
    }
}

#[test]
fn test_zero_size_tensor() {
    let path = test_dir().join("zero_size.axon");
    let mut builder = AxonBuilder::new();
    let data: Vec<u8> = vec![];
    builder = builder.add_tensor("empty", data, DType::U8, &[0]);
    let bytes = builder.build().unwrap();
    write_raw(&path, &bytes);
    let rt = AxonRuntime::open(&path).unwrap();
    let view = rt.tensor_view("empty").unwrap();
    assert_eq!(view.len(), 0);
}

#[test]
fn test_many_tensors() {
    let path = test_dir().join("many_tensors.axon");
    let mut builder = AxonBuilder::new();
    for i in 0..1000 {
        builder = builder.add_tensor(&format!("tensor_{:04}", i), vec![i as u8; 4], DType::U8, &[4]);
    }
    let bytes = builder.build().unwrap();
    write_raw(&path, &bytes);
    let rt = AxonRuntime::open(&path).unwrap();
    assert_eq!(rt.tensor_count(), 1000);
}
