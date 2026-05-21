//! Axon runtime benchmark — measures open time, tensor access, memory usage.
//!
//! Usage:
//!   cargo run --release --example bench_axon -- model.axon
//!   or
//!   ./target/release/examples/bench_axon model.axon

use std::fs;
use std::time::Instant;
use axon_runtime::AxonRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_axon <model.axon>");
        std::process::exit(1);
    }

    let path = &args[1];
    let file_size = fs::metadata(path)?.len();

    println!("=== Axon Runtime Benchmark ===");
    println!("File: {}", path);
    println!("File size: {:.2} MB", file_size as f64 / (1024.0 * 1024.0));

    // ── Open ──
    let t0 = Instant::now();
    let rt = AxonRuntime::open(path)?;
    let open_time = t0.elapsed();
    println!("\n[Open] {:.2?}", open_time);
    println!("  Model: {}", rt.model_name());
    println!("  Architecture: {}", rt.architecture());
    println!("  Tensors: {}", rt.tensor_count());

    // ── First tensor access ──
    let names = rt.tensor_names();
    let first = names[0];
    let t0 = Instant::now();
    let _view = rt.tensor_view(first)?;
    let first_access = t0.elapsed();
    println!("\n[First tensor access] {} ({})", first, first_access.as_secs_f64() * 1000.0);

    // ── Random tensor access ──
    let idx = names.len() / 2;
    let mid = names[idx];
    let t0 = Instant::now();
    let _view = rt.tensor_view(mid)?;
    let random_access = t0.elapsed();
    println!("\n[Random tensor access] {} ({})", mid, random_access.as_secs_f64() * 1000.0);

    // ── 100 random accesses ──
    let indices: Vec<usize> = (0..100.min(names.len())).collect();
    let t0 = Instant::now();
    for &i in &indices {
        let _ = rt.tensor_view(names[i])?;
    }
    let hundred_access = t0.elapsed();
    println!("\n[100 tensor accesses] {:.2?} ({:.3?} avg)", hundred_access, hundred_access / 100);

    // ── Full tensor scan ──
    let t0 = Instant::now();
    for name in &names {
        let _ = rt.tensor_view(name)?;
    }
    let full_scan = t0.elapsed();
    let total = vtensor_total_bytes(&rt);
    let throughput = total as f64 / full_scan.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!("\n[Full tensor scan] {:.2?}", full_scan);
    println!("  Total accessed: {:.2} MB", total as f64 / (1024.0 * 1024.0));
    println!("  Throughput: {:.2} GB/s", throughput);

    // ── Partial tensor access ──
    if names.len() > 1 {
        let name = names[names.len() / 2];
        if let Ok(info) = rt.tensor_info(name) {
            let slice_size = (info.data_size / 4).min(4096);
            let t0 = Instant::now();
            let _view = rt.tensor_byte_view(name, 0..slice_size as usize)?;
            let partial_time = t0.elapsed();
            println!("\n[Partial tensor access] {:.2?} ({} bytes)", partial_time, slice_size);
        }
    }

    // ── Row access (2D tensors only) ──
    for name in &names {
        if let Ok(info) = rt.tensor_info(name) {
            if info.shape.len() == 2 && info.shape[0] > 2 {
                let t0 = Instant::now();
                let _rows = rt.tensor_rows(name, 0, 2)?;
                let row_time = t0.elapsed();
                println!("\n[Row access] {} (rows 0..2 of {}x{})", name, info.shape[0], info.shape[1]);
                println!("  Time: {:.2?}", row_time);
                break;
            }
        }
    }

    // ── Stats ──
    let stats = rt.stats();
    println!("\n[Runtime stats]");
    println!("  Bytes read: {}", stats.bytes_read());
    println!("  Access count: {}", stats.tensor_accesses());

    println!("\n=== Done ===");
    Ok(())
}

fn vtensor_total_bytes(rt: &AxonRuntime) -> u64 {
    rt.tensors().iter().map(|t| t.data_size).sum()
}
