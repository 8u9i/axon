use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn axon_bin() -> &'static str {
    env!("CARGO_BIN_EXE_axon")
}

fn test_workspace(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "axon_cli_{name}_{}_{}",
        std::process::id(),
        unique_suffix()
    ));
    fs::create_dir_all(&path).expect("create test workspace");
    path
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos()
}

fn run(args: &[&str]) -> Output {
    Command::new(axon_bin())
        .args(args)
        .output()
        .expect("run axon binary")
}

fn assert_success(output: Output, context: &str) {
    if !output.status.success() {
        panic!(
            "{context} failed\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn assert_failure(output: Output, context: &str) {
    if output.status.success() {
        panic!(
            "{context} unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn write_manifest(path: &Path) {
    let manifest = r#"{
  "tensors": [
    { "name": "tiny_weight", "dtype": 5, "shape": [16] },
    { "name": "matrix", "dtype": 0, "shape": [2, 2] }
  ]
}"#;
    fs::write(path, manifest).expect("write manifest");
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn write_tiny_gguf(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 1); // tensor count
    push_u64(&mut bytes, 2); // metadata kv count

    push_gguf_string(&mut bytes, "general.name");
    push_u32(&mut bytes, 8); // string
    push_gguf_string(&mut bytes, "TinyGGUF");

    push_gguf_string(&mut bytes, "general.architecture");
    push_u32(&mut bytes, 8); // string
    push_gguf_string(&mut bytes, "test");

    push_gguf_string(&mut bytes, "matrix");
    push_u32(&mut bytes, 2); // rank
    push_u64(&mut bytes, 2);
    push_u64(&mut bytes, 2);
    push_u32(&mut bytes, 0); // GGML_TYPE_F32
    push_u64(&mut bytes, 0); // tensor offset relative to data section

    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&[
        0, 0, 128, 63, // 1.0
        0, 0, 0, 64, // 2.0
        0, 0, 64, 64, // 3.0
        0, 0, 128, 64, // 4.0
    ]);

    fs::write(path, bytes).expect("write tiny gguf");
}

fn write_fake_ollama_model_store(root: &Path, model_ref: &str) -> PathBuf {
    let (model, tag) = model_ref.rsplit_once(':').unwrap_or((model_ref, "latest"));
    let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let manifest_dir = root
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join(model);
    let blob_dir = root.join("blobs");
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&blob_dir).expect("create blob dir");

    let blob_path = blob_dir.join(format!("sha256-{digest}"));
    write_tiny_gguf(&blob_path);

    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": format!("sha256:{digest}"),
                "size": fs::metadata(&blob_path).expect("blob metadata").len()
            }
        ]
    });
    fs::write(
        manifest_dir.join(tag),
        serde_json::to_vec(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");
    blob_path
}

#[test]
fn pack_validate_list_extract_and_runtime_inspect() {
    let dir = test_workspace("happy_path");
    let manifest = dir.join("manifest.json");
    let data_dir = dir.join("data");
    let model = dir.join("model.axon");
    let extracted = dir.join("tiny_weight.bin");
    fs::create_dir_all(&data_dir).expect("create data dir");
    write_manifest(&manifest);

    assert_success(
        run(&[
            "pack",
            "--manifest",
            &path_string(&manifest),
            "--data-dir",
            &path_string(&data_dir),
            "--output",
            &path_string(&model),
            "--model",
            "TinySmoke",
            "--architecture",
            "test",
        ]),
        "pack",
    );
    assert_success(run(&["validate", &path_string(&model)]), "validate");

    let list = run(&["list", &path_string(&model), "--verbose"]);
    assert_success(list, "list");

    let inspect = run(&["inspect", &path_string(&model)]);
    assert_success(inspect, "inspect");

    assert_success(
        run(&[
            "extract",
            &path_string(&model),
            "--name",
            "tiny_weight",
            "--output",
            &path_string(&extracted),
        ]),
        "extract",
    );
    assert_eq!(fs::metadata(&extracted).expect("extracted file").len(), 16);

    assert_success(
        run(&["runtime", "inspect", &path_string(&model)]),
        "runtime inspect",
    );
    assert_success(
        run(&["runtime", "tensor", &path_string(&model), "tiny_weight"]),
        "runtime tensor",
    );
    assert_success(
        run(&[
            "runtime",
            "slice",
            &path_string(&model),
            "matrix",
            "--rows",
            "0,1",
        ]),
        "runtime row slice",
    );
    assert_success(
        run(&[
            "runtime",
            "slice",
            &path_string(&model),
            "tiny_weight",
            "--bytes",
            "0,8",
        ]),
        "runtime byte slice",
    );

    let converted = dir.join("manifest-out.json");
    assert_success(
        run(&["convert", &path_string(&model), &path_string(&converted)]),
        "convert",
    );
    let converted_json = fs::read_to_string(&converted).expect("read converted manifest");
    assert!(converted_json.contains("TinySmoke"));

    let unpacked = dir.join("unpacked");
    assert_success(
        run(&[
            "unpack",
            &path_string(&model),
            "--output",
            &path_string(&unpacked),
        ]),
        "unpack",
    );
    assert!(unpacked.join("tiny_weight.npy").exists());

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn invalid_input_fails_without_panic_banner() {
    let dir = test_workspace("invalid_input");
    let invalid = dir.join("not_an_axon.txt");
    fs::write(&invalid, "not an axon file").expect("write invalid input");

    let output = run(&["inspect", &path_string(&invalid)]);
    assert_failure(output, "inspect invalid input");

    let output = run(&["inspect", &path_string(&invalid)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error: failed to parse"));
    assert!(!stderr.contains("panicked at"));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_tensor_reports_clean_error() {
    let dir = test_workspace("missing_tensor");
    let manifest = dir.join("manifest.json");
    let data_dir = dir.join("data");
    let model = dir.join("model.axon");
    let extracted = dir.join("missing.bin");
    fs::create_dir_all(&data_dir).expect("create data dir");
    write_manifest(&manifest);

    assert_success(
        run(&[
            "pack",
            "--manifest",
            &path_string(&manifest),
            "--data-dir",
            &path_string(&data_dir),
            "--output",
            &path_string(&model),
        ]),
        "pack",
    );

    let output = run(&[
        "extract",
        &path_string(&model),
        "--name",
        "does_not_exist",
        "--output",
        &path_string(&extracted),
    ]);
    assert_failure(output, "extract missing tensor");

    let output = run(&[
        "extract",
        &path_string(&model),
        "--name",
        "does_not_exist",
        "--output",
        &path_string(&extracted),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error: tensor 'does_not_exist' not found"));
    assert!(!stderr.contains("panicked at"));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn import_tiny_gguf_then_inspect_with_axon() {
    let dir = test_workspace("gguf_import");
    let gguf = dir.join("tiny.gguf");
    let axon = dir.join("tiny.axon");
    write_tiny_gguf(&gguf);

    assert_success(
        run(&[
            "import-gguf",
            &path_string(&gguf),
            "--output",
            &path_string(&axon),
        ]),
        "import-gguf",
    );

    let inspect = run(&["inspect", &path_string(&axon)]);
    assert_success(inspect, "inspect imported gguf");
    let list = run(&["list", &path_string(&axon), "--verbose"]);
    assert_success(list, "list imported gguf");
    assert_success(
        run(&["validate", &path_string(&axon)]),
        "validate imported gguf",
    );
    assert_success(
        run(&["runtime", "tensor", &path_string(&axon), "matrix"]),
        "runtime tensor imported gguf",
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn import_ollama_model_store_then_inspect_with_axon() {
    let dir = test_workspace("ollama_import");
    let store = dir.join("ollama-models");
    let axon = dir.join("gemma.axon");
    let blob = write_fake_ollama_model_store(&store, "gemma3:1b");

    assert_success(
        run(&[
            "import-ollama",
            "gemma3:1b",
            "--models-dir",
            &path_string(&store),
            "--output",
            &path_string(&axon),
        ]),
        "import-ollama",
    );

    assert_success(
        run(&["inspect", &path_string(&axon)]),
        "inspect ollama import",
    );
    assert_success(
        run(&["runtime", "tensor", &path_string(&axon), "matrix"]),
        "runtime tensor ollama import",
    );
    assert!(blob.exists());

    fs::remove_dir_all(&dir).ok();
}
