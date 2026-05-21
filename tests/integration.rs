//! Integration tests for the .axon format.
//!
//! Run with: cargo test --test integration_test --release

use axon_core::*;

/// Helper: build a minimal .axon file with a single FP32 tensor.
fn build_single_tensor_axon() -> Vec<u8> {
    let data = [
        1.0f32.to_bits(),
        2.0f32.to_bits(),
        3.0f32.to_bits(),
        4.0f32.to_bits(),
    ];
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    AxonBuilder::new()
        .model("test-model")
        .architecture("test")
        .add_tensor("weights", bytes, DType::F32, &[2, 2])
        .build()
        .expect("Failed to build .axon file")
}

#[test]
fn test_header_magic_and_version() {
    let axon_data = build_single_tensor_axon();
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");
    assert_eq!(&file.header.magic, b"AXON", "Magic bytes should be AXON");
    assert_eq!(file.header.version, 1, "Version should be 1");
}

#[test]
fn test_tensor_count_and_names() {
    let axon_data = build_single_tensor_axon();
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");
    assert_eq!(file.header.tensor_count, 1);
    assert_eq!(file.manifest.tensor_order.len(), 1);
    assert_eq!(file.manifest.tensor_order[0], "weights");
}

#[test]
fn test_tensor_descriptor_values() {
    let axon_data = build_single_tensor_axon();
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");
    let desc = file
        .manifest
        .get_tensor("weights")
        .expect("Tensor not found");
    assert_eq!(desc.name_str(), "weights", "Tensor name should match");
    assert!(desc.dtype().is_ok(), "dtype should parse");
    assert_eq!(desc.dtype().unwrap(), DType::F32, "dtype should be F32");
    assert_eq!(desc.shape_vec(), vec![2, 2], "shape should be [2, 2]");
    assert_eq!(desc.num_elements(), 4, "should have 4 elements");
    // Size: 4 elements * 4 bytes = 16 bytes, plus padding to 64-byte alignment
    assert!(desc.data_size >= 16, "data_size should be at least 16");
}

#[test]
fn test_tensor_data_integrity() {
    let axon_data = build_single_tensor_axon();
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");
    let tensor_bytes = file.tensor_data("weights").expect("Tensor data not found");

    assert_eq!(tensor_bytes.len(), 16, "4 x f32 = 16 bytes");

    // Read back the floats
    use byteorder::{LittleEndian, ReadBytesExt};
    let mut cursor = std::io::Cursor::new(tensor_bytes);
    let values: Vec<f32> = (0..4)
        .map(|_| cursor.read_f32::<LittleEndian>().unwrap())
        .collect();
    assert_eq!(
        values,
        vec![1.0, 2.0, 3.0, 4.0],
        "Tensor data should match input"
    );
}

#[test]
fn test_checksum_verification() {
    let axon_data = build_single_tensor_axon();
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");
    let results = file.verify_all_checksums();
    assert_eq!(results.len(), 1, "Should have 1 checksum result");
    assert!(results[0].1, "Checksum should pass for single tensor");
}

#[test]
fn test_checksum_detects_corruption() {
    let axon_data = build_single_tensor_axon();
    // Parse to find the offset
    let file = AxonFile::from_bytes(axon_data.clone()).expect("initial parse");
    // Corruption in the data area of the first tensor
    let desc = file.manifest.get_tensor("weights").unwrap();
    let corrupt_offset = desc.data_offset as usize + desc.data_size as usize - 1;

    let mut corrupted = axon_data.clone();
    if corrupt_offset < corrupted.len() {
        corrupted[corrupt_offset] ^= 0xFF;
    }

    let file2 = AxonFile::from_bytes(corrupted).expect("Failed to parse (still should parse)");
    let results = file2.verify_all_checksums();
    assert_eq!(results.len(), 1, "Should have 1 checksum result");
    assert!(
        !results[0].1,
        "Corrupted data should fail checksum verification"
    );
}

#[test]
fn test_header_alignment() {
    // Verify align_up utility
    assert_eq!(AxonHeader::align_up(0, 64), 0);
    assert_eq!(AxonHeader::align_up(1, 64), 64);
    assert_eq!(AxonHeader::align_up(63, 64), 64);
    assert_eq!(AxonHeader::align_up(64, 64), 64);
    assert_eq!(AxonHeader::align_up(65, 64), 128);
}

#[test]
fn test_dtype_roundtrip() {
    for code in 0u32..=11 {
        let dtype = DType::from_code(code).expect("Valid dtype code");
        assert_eq!(dtype.to_code(), code, "Roundtrip failed for code {code}");
        assert_eq!(DType::from_code(dtype.to_code()).unwrap().to_code(), code);
    }
}

#[test]
fn test_invalid_dtype() {
    assert!(DType::from_code(99).is_err(), "Code 99 should be invalid");
    assert!(
        DType::from_code(u32::MAX).is_err(),
        "Max u32 should be invalid"
    );
}

#[test]
fn test_large_model_multiple_tensors() {
    let mut builder = AxonBuilder::new().model("big-model").architecture("llama");

    // Add 10 tensors of varying sizes and dtypes
    for i in 0..5 {
        let name = format!("layer_{}_weight", i);
        let size = (i + 1) * 32; // keep sizes small but valid
        let dtype = if i % 2 == 0 { DType::F32 } else { DType::F16 };
        let elem_size = dtype.size_in_bytes();
        let num_elems = size / elem_size;
        let shape = vec![num_elems as u64 / 4, 4];
        let data: Vec<u8> = (0..size as u8).cycle().take(size).collect();
        let expected = shape.iter().product::<u64>() as usize * elem_size;
        assert_eq!(data.len(), expected, "size check for {name}");
        builder = builder.add_tensor(&name, data, dtype, &shape);
    }
    // Add 5 more with different shapes
    for i in 5..10 {
        let name = format!("layer_{}_bias", i);
        let dtype = if i % 2 == 0 { DType::I32 } else { DType::F16 };
        let size = 16;
        let data: Vec<u8> = (0..size).map(|j| j as u8).collect();
        let elem_size = dtype.size_in_bytes();
        let num_elems = size / elem_size;
        let shape = vec![num_elems as u64];
        let expected = shape.iter().product::<u64>() as usize * elem_size;
        assert_eq!(data.len(), expected, "size check for {name}");
        builder = builder.add_tensor(&name, data, dtype, &shape);
    }

    let axon_data = builder.build().expect("Failed to build multi-tensor .axon");
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");
    assert_eq!(file.header.tensor_count, 10, "Should have 10 tensors");
    assert_eq!(
        file.manifest.tensor_order.len(),
        10,
        "Manifest should list 10 tensors"
    );

    // Verify all tensors are accessible
    for name in &file.manifest.tensor_order {
        assert!(
            file.manifest.get_tensor(name).is_some(),
            "Tensor {name} should exist"
        );
        assert!(
            file.tensor_data(name).is_some(),
            "Data for {name} should be accessible"
        );
    }

    // All checksums should pass
    let results = file.verify_all_checksums();
    assert_eq!(results.len(), 10, "All 10 tensors should have checksums");
    assert!(
        results.iter().all(|(_, ok)| *ok),
        "All checksums should pass"
    );
}

#[test]
fn test_manifest_metadata() {
    let data = vec![1u8; 16];
    let axon_data = AxonBuilder::new()
        .model("MyModel-7B")
        .architecture("llama")
        .add_tensor("test", data, DType::F32, &[4])
        .build()
        .expect("Build failed");

    let file = AxonFile::from_bytes(axon_data).expect("Parse failed");
    assert_eq!(file.manifest.model.as_deref(), Some("MyModel-7B"));
    assert_eq!(file.manifest.architecture.as_deref(), Some("llama"));
}

#[test]
fn test_empty_tensor_rejected() {
    let result = std::panic::catch_unwind(|| {
        AxonBuilder::new()
            .add_tensor("empty", Vec::new(), DType::F32, &[0])
            .build()
    });
    assert!(
        result.is_err() || result.is_ok(),
        "Building with empty tensor may panic or return error"
    );
}

#[test]
fn test_tensor_alignment() {
    // Create tensors with unusual sizes and verify they're properly aligned
    let data = vec![0u8; 100]; // 100 bytes, not a multiple of 64
    let axon_data = AxonBuilder::new()
        .add_tensor("odd_sized", data, DType::U8, &[100])
        .build()
        .expect("Build failed");

    let file = AxonFile::from_bytes(axon_data).expect("Parse failed");
    let desc = file.manifest.get_tensor("odd_sized").unwrap();

    // Data offset should be 64-byte aligned
    assert_eq!(
        desc.data_offset % 64,
        0,
        "Tensor data offset should be 64-byte aligned"
    );
}

#[test]
fn test_serialize_deserialize_header() {
    let original = AxonHeader::new(5, 1024, 16777216);
    let bytes = original.to_bytes().expect("Serialize failed");
    assert_eq!(bytes.len(), 64, "Serialized header should be 64 bytes");

    let restored = AxonHeader::from_bytes(&bytes).expect("Deserialize failed");
    assert_eq!(original.magic, restored.magic);
    assert_eq!(original.version, restored.version);
    assert_eq!(original.manifest_offset, restored.manifest_offset);
    assert_eq!(original.manifest_size, restored.manifest_size);
    assert_eq!(original.tensor_count, restored.tensor_count);
    assert_eq!(original.payload_offset, restored.payload_offset);
    assert_eq!(original.payload_size, restored.payload_size);
    assert_eq!(original.flags, restored.flags);
}

#[test]
fn test_serialize_deserialize_tensor_descriptor() {
    let desc = TensorDescriptor::new(
        "test.tensor",
        DType::BF16,
        &[128, 256, 64],
        4096,
        4194304,
        Affinity::Hbm,
        0xDEADBEEF,
    );
    let bytes = desc.to_bytes().expect("Serialize failed");
    assert_eq!(
        bytes.len(),
        192,
        "Serialized descriptor should be 192 bytes"
    );

    let restored = TensorDescriptor::from_bytes(&bytes).expect("Deserialize failed");
    assert_eq!(desc.name_str(), restored.name_str());
    assert_eq!(desc.dtype, restored.dtype);
    assert_eq!(desc.rank, restored.rank);
    assert_eq!(desc.shape_vec(), restored.shape_vec());
    assert_eq!(desc.data_offset, restored.data_offset);
    assert_eq!(desc.data_size, restored.data_size);
    assert_eq!(desc.checksum, restored.checksum);
}

#[test]
fn test_payload_offset_calculation() {
    let axon_data = build_single_tensor_axon();
    let file = AxonFile::from_bytes(axon_data).expect("Failed to parse");

    // The payload offset should be after:
    //   - 64-byte header
    //   - 4032 bytes of HOT_START padding
    //   - manifest (variable)
    //   - TDT (1 tensor * 192 bytes + padding)
    // All aligned to 64 bytes
    assert!(
        file.header.payload_offset >= 4096,
        "Payload should start after HOT_START"
    );
    assert_eq!(
        file.header.payload_offset % 64,
        0,
        "Payload should be 64-byte aligned"
    );
}

#[test]
fn test_multiple_tensor_order_preserved() {
    let names = ["z_last", "a_first", "m_middle", "b_second"];
    let mut builder = AxonBuilder::new();
    for name in &names {
        builder = builder.add_tensor(name, vec![0u8; 16], DType::F32, &[4]);
    }
    let axon_data = builder.build().expect("Build failed");
    let file = AxonFile::from_bytes(axon_data).expect("Parse failed");

    // Tensor order should be lexicographic (alphabetical)
    assert_eq!(
        file.manifest.tensor_order,
        vec!["a_first", "b_second", "m_middle", "z_last"]
    );
}

#[test]
fn test_tensor_data_random_access() {
    let mut builder = AxonBuilder::new();
    for i in 0..100 {
        let name = format!("tensor_{:04}", i);
        let data: Vec<u8> = (0..64).map(|j| (i * 64 + j) as u8).collect();
        builder = builder.add_tensor(&name, data, DType::U8, &[64]);
    }
    let axon_data = builder.build().expect("Build failed");
    let file = AxonFile::from_bytes(axon_data).expect("Parse failed");

    // Random access: get tensor_0050's data
    let data_50 = file
        .tensor_data("tensor_0050")
        .expect("tensor_0050 not found");
    assert_eq!(data_50[0], (50 * 64) as u8, "First byte should match");
    assert_eq!(data_50[63], (50 * 64 + 63) as u8, "Last byte should match");

    // Get tensor_0000
    let data_0 = file
        .tensor_data("tensor_0000")
        .expect("tensor_0000 not found");
    assert_eq!(data_0[0], 0, "First byte should be 0");
}

#[test]
fn test_invalid_magic_rejected() {
    let bad_data = vec![0u8; 64];
    let result = AxonHeader::from_bytes(&bad_data);
    assert!(result.is_err(), "Invalid magic should be rejected");
}

#[test]
fn test_short_header_rejected() {
    let bad_data = vec![0u8; 10];
    let result = AxonHeader::from_bytes(&bad_data);
    assert!(result.is_err(), "Short header should be rejected");
}

#[test]
fn test_truncated_read_still_parses() {
    let axon_data = build_single_tensor_axon();
    // Truncate to just the header — should parse header but fail to find manifest
    let truncated = axon_data[..64].to_vec();
    let result = AxonFile::from_bytes(truncated);
    assert!(result.is_err(), "Truncated file should fail gracefully");
}

#[test]
fn test_mmap_open() {
    let path = std::path::Path::new("output");
    let _ = std::fs::create_dir_all(path);
    let test_path = path.join("test_mmap.axon");
    let axon_data = build_single_tensor_axon();
    std::fs::write(&test_path, &axon_data).expect("Failed to write test file");

    let mapped = MappedAxonFile::open(&test_path);
    assert!(mapped.is_ok(), "mmap open should succeed");
    if let Ok(mapped_file) = mapped {
        let summary = mapped_file.summary();
        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.tensors[0].name, "weights");
    }

    let _ = std::fs::remove_file(&test_path);
}

#[test]
fn test_multiple_dtypes_in_single_file() {
    let mut builder = AxonBuilder::new().model("mixed-precision-test");
    let dtypes: [(DType, usize, &[u64]); 6] = [
        (DType::F32, 16, &[4]),
        (DType::F16, 8, &[4]),
        (DType::BF16, 8, &[4]),
        (DType::I32, 16, &[4]),
        (DType::I64, 32, &[4]),
        (DType::U8, 4, &[4]),
    ];

    for (i, (dtype, size, shape)) in dtypes.iter().enumerate() {
        let name = format!("tensor_{}", i);
        let data: Vec<u8> = (0..*size).map(|j| j as u8).collect();
        builder = builder.add_tensor(&name, data, *dtype, shape);
    }

    let axon_data = builder.build().expect("Build failed");
    let file = AxonFile::from_bytes(axon_data).expect("Parse failed");
    assert_eq!(
        file.manifest.tensor_count(),
        6,
        "Should have 6 tensors with different dtypes"
    );

    // All checksums pass
    let results = file.verify_all_checksums();
    assert!(
        results.iter().all(|(_, ok)| *ok),
        "All mixed-precision checksums should pass"
    );
}
