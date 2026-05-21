//! # Axon Runtime Benchmarks
//!
//! Measures the key performance characteristics of the runtime:
//!
//! - **Open time**: time to parse header + manifest + TDT (no tensor data touched)
//! - **First tensor access**: time to read the first tensor from mmap
//! - **Random tensor access**: time to access 10 randomly selected tensors
//! - **Sequential access**: time to iterate all tensors
//! - **Byte range vs full load**: time/memory for partial access vs full tensor
//!
//! ## Usage
//!
//! ```bash
//! # Generate test files first
//! cargo build --release
//! ./target/release/axon create --model "bench-model" output/bench.axon
//!
//! # Run benchmarks
//! cargo bench --package axon-runtime
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use axon_core::{AxonFile, AxonBuilder, DType};
use axon_runtime::AxonRuntime;

fn test_dir() -> PathBuf {
    let dir = PathBuf::from("output");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Generate a synthetic .axon file with N tensors, each of `tensor_size` bytes.
fn generate_model(path: &PathBuf, tensor_count: usize, tensor_size: usize) {
    let data: Vec<u8> = (0..tensor_size).map(|i| i as u8).collect();
    let mut builder = AxonBuilder::new().model("bench-model").architecture("bench");

    for i in 0..tensor_count {
        let name = format!("layer_{}_weight", i);
        let shape = vec![tensor_size as u64];
        builder = builder.add_tensor(&name, data.clone(), DType::U8, &shape);
    }

    let bytes = builder.build().expect("Failed to build .axon");
    fs::write(path, &bytes).expect("Failed to write .axon");
}

// ── Helpers for measurement ────────────────────────────────────────

fn bench_open(path: &PathBuf) -> Duration {
    let start = Instant::now();
    let _rt = AxonRuntime::open(path).expect("Failed to open");
    start.elapsed()
}

fn bench_open_fallback(path: &PathBuf) -> Duration {
    let data = fs::read(path).expect("Failed to read");
    let start = Instant::now();
    let _file = AxonFile::from_bytes(data).expect("Failed to parse");
    start.elapsed()
}

fn bench_first_tensor(rt: &AxonRuntime) -> Duration {
    let start = Instant::now();
    let _data = rt.tensor("layer_0_weight").expect("Failed to get tensor");
    start.elapsed()
}

fn bench_first_tensor_fallback(file: &AxonFile) -> Duration {
    let start = Instant::now();
    let _data = file.tensor_data("layer_0_weight").expect("Failed to get tensor");
    start.elapsed()
}

#[allow(dead_code)]
fn bench_random_access(rt: &AxonRuntime, names: &[&str]) -> Duration {
    let start = Instant::now();
    for name in names {
        let _data = rt.tensor(name).expect("Failed to get tensor");
    }
    start.elapsed()
}

fn bench_all_tensors(rt: &AxonRuntime) -> Duration {
    let names = rt.tensor_names();
    let start = Instant::now();
    for name in &names {
        let _data = rt.tensor(name).expect("Failed to get tensor");
    }
    start.elapsed()
}

fn bench_byte_range(rt: &AxonRuntime, name: &str, offset: u64, size: u64) -> Duration {
    let start = Instant::now();
    let _data = rt.tensor_byte_range(name, offset, size).expect("Failed to get byte range");
    start.elapsed()
}

fn bench_full_tensor(rt: &AxonRuntime, name: &str) -> Duration {
    let start = Instant::now();
    let _data = rt.tensor(name).expect("Failed to get tensor");
    start.elapsed()
}

// ── Test configurations ────────────────────────────────────────────

const SMALL_COUNT: usize = 10;
const SMALL_SIZE: usize = 1024;       // 1KB per tensor, ~10MB total
const MEDIUM_COUNT: usize = 100;
const MEDIUM_SIZE: usize = 1024 * 32; // 32KB per tensor, ~3.2MB total
const LARGE_COUNT: usize = 100;
const LARGE_SIZE: usize = 1024 * 1024; // 1MB per tensor, ~100MB total

fn main() {
    println!("=== Axon Runtime Benchmarks ===\n");
    println!("Generating test files...\n");

    let small_path = test_dir().join("bench_small.axon");
    generate_model(&small_path, SMALL_COUNT, SMALL_SIZE);

    let medium_path = test_dir().join("bench_medium.axon");
    generate_model(&medium_path, MEDIUM_COUNT, MEDIUM_SIZE);

    let large_path = test_dir().join("bench_large.axon");
    generate_model(&large_path, LARGE_COUNT, LARGE_SIZE);

    println!("Test files ready:");
    println!("  Small:  {} tensors, {} bytes each", SMALL_COUNT, SMALL_SIZE);
    println!("  Medium: {} tensors, {} bytes each", MEDIUM_COUNT, MEDIUM_SIZE);
    println!("  Large:  {} tensors, {} bytes each", LARGE_COUNT, LARGE_SIZE);
    println!();

    // ── Benchmark 1: Open time ────────────────────────────────────
    println!("---");
    println!("## 1. Open Time\n");

    // Warm up each file by opening once
    let _ = AxonRuntime::open(&small_path).unwrap();
    let _ = AxonRuntime::open(&medium_path).unwrap();
    let _ = AxonRuntime::open(&large_path).unwrap();

    const ITERATIONS: u32 = 100;

    // Small model
    let mut times = Vec::new();
    for _ in 0..ITERATIONS {
        times.push(bench_open(&small_path));
    }
    let avg = times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Runtime | Small ({} tensors) | Open time | {:>12?} |", SMALL_COUNT, avg);

    // Medium model
    let mut times = Vec::new();
    for _ in 0..ITERATIONS {
        times.push(bench_open(&medium_path));
    }
    let avg = times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Runtime | Medium ({} tensors) | Open time | {:>12?} |", MEDIUM_COUNT, avg);

    // Large model
    let mut times = Vec::new();
    for _ in 0..ITERATIONS {
        times.push(bench_open(&large_path));
    }
    let avg = times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Runtime | Large ({} tensors) | Open time | {:>12?} |", LARGE_COUNT, avg);

    // ── Benchmark 2: Runtime open vs fallback ─────────────────────
    println!("\n---");
    println!("## 2. Open: Runtime (mmap) vs Fallback (eager Vec)\n");

    // Small
    let mut rt_times = Vec::new();
    let mut fb_times = Vec::new();
    for _ in 0..ITERATIONS {
        rt_times.push(bench_open(&small_path));
        fb_times.push(bench_open_fallback(&small_path));
    }
    let rt_avg = rt_times.iter().sum::<Duration>() / ITERATIONS;
    let fb_avg = fb_times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Small  | Runtime (mmap) | {:>12?} |", rt_avg);
    println!("| Small  | Fallback (eager) | {:>12?} |", fb_avg);

    // Medium
    let mut rt_times = Vec::new();
    let mut fb_times = Vec::new();
    for _ in 0..ITERATIONS {
        rt_times.push(bench_open(&medium_path));
        fb_times.push(bench_open_fallback(&medium_path));
    }
    let rt_avg = rt_times.iter().sum::<Duration>() / ITERATIONS;
    let fb_avg = fb_times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Medium | Runtime (mmap) | {:>12?} |", rt_avg);
    println!("| Medium | Fallback (eager) | {:>12?} |", fb_avg);

    // Large
    let mut rt_times = Vec::new();
    let mut fb_times = Vec::new();
    for _ in 0..ITERATIONS {
        rt_times.push(bench_open(&large_path));
        fb_times.push(bench_open_fallback(&large_path));
    }
    let rt_avg = rt_times.iter().sum::<Duration>() / ITERATIONS;
    let fb_avg = fb_times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Large  | Runtime (mmap) | {:>12?} |", rt_avg);
    println!("| Large  | Fallback (eager) | {:>12?} |", fb_avg);

    // ── Benchmark 3: First tensor access ──────────────────────────
    println!("\n---");
    println!("## 3. First Tensor Access Time\n");

    let rt = AxonRuntime::open(&small_path).unwrap();
    let file_data = fs::read(&small_path).unwrap();
    let file = AxonFile::from_bytes(file_data).unwrap();

    let mut rt_times = Vec::new();
    let mut fb_times = Vec::new();
    for _ in 0..ITERATIONS {
        rt_times.push(bench_first_tensor(&rt));
        fb_times.push(bench_first_tensor_fallback(&file));
    }
    let rt_avg = rt_times.iter().sum::<Duration>() / ITERATIONS;
    let fb_avg = fb_times.iter().sum::<Duration>() / ITERATIONS;
    println!("| Runtime (mmap) | {:>12?} |", rt_avg);
    println!("| Fallback (eager) | {:>12?} |", fb_avg);

    // ── Benchmark 4: Tensor slicing — partial vs full load ────────
    println!("\n---");
    println!("## 4. Partial Load vs Full Load (100MB model)\n");

    let rt_large = AxonRuntime::open(&large_path).unwrap();

    let mut full_times = Vec::new();
    let mut partial_times = Vec::new();
    for _ in 0..ITERATIONS {
        full_times.push(bench_full_tensor(&rt_large, "layer_0_weight"));
        partial_times.push(bench_byte_range(&rt_large, "layer_0_weight", 0, 4096));
    }
    let full_avg = full_times.iter().sum::<Duration>() / ITERATIONS;
    let partial_avg = partial_times.iter().sum::<Duration>() / ITERATIONS;

    // Note: on first access, time is dominated by OS page fault + file system cache.
    // The benchmark is run after warm-up, so subsequent accesses hit the page cache.
    println!("| Full tensor (1MB) | {:>12?} |", full_avg);
    println!("| First 4KB only    | {:>12?} |", partial_avg);

    // ── Benchmark 5: Sequential access ────────────────────────────
    println!("\n---");
    println!("## 5. Sequential Access (all tensors)\n");

    let rt_med = AxonRuntime::open(&medium_path).unwrap();
    let mut seq_times = Vec::new();
    for _ in 0..ITERATIONS.min(10) {
        seq_times.push(bench_all_tensors(&rt_med));
    }
    let seq_avg = seq_times.iter().sum::<Duration>() / seq_times.len() as u32;
    println!("| Runtime | {} tensors ({}MB total) | {:>12?} |",
             MEDIUM_COUNT, (MEDIUM_COUNT * MEDIUM_SIZE) / (1024 * 1024), seq_avg);

    // ── Summary ───────────────────────────────────────────────────
    println!("\n---");
    println!("## Summary\n");

    let small_meta = fs::metadata(&small_path).unwrap();
    let medium_meta = fs::metadata(&medium_path).unwrap();
    let large_meta = fs::metadata(&large_path).unwrap();

    println!("| File | Size | Tensors | Tensor Size |");
    println!("|------|------|---------|-------------|");
    println!("| Small  | {} bytes | {} | {} bytes |", small_meta.len(), SMALL_COUNT, SMALL_SIZE);
    println!("| Medium | {} bytes | {} | {} bytes |", medium_meta.len(), MEDIUM_COUNT, MEDIUM_SIZE);
    println!("| Large  | {} bytes | {} | {} bytes |", large_meta.len(), LARGE_COUNT, LARGE_SIZE);
}
