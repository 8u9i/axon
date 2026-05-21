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
