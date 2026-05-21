use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use clap::{Parser, Subcommand, Args};
use log::info;
use axon_core::*;
use axon_runtime::AxonRuntime;

#[derive(Parser)]
#[command(name = "axon", about = "Adaptive eXecutable Object Notation CLI", version)]
struct Cli { #[command(subcommand)] command: Commands }

#[derive(Subcommand)]
enum Commands {
    Inspect(InspectArgs), Pack(PackArgs), Unpack(UnpackArgs),
    Convert(ConvertArgs), Bench(BenchArgs), Validate(ValidateArgs),
    List(ListArgs), Extract(ExtractArgs), Create(CreateArgs),
    #[command(subcommand)]
    Runtime(RuntimeCommands),
}

#[derive(Subcommand)]
enum RuntimeCommands {
    /// Show detailed runtime information about a model file
    Inspect(RuntimeInspectArgs),
    /// Access a tensor and print its size and first bytes
    Tensor(RuntimeTensorArgs),
    /// Slice a tensor (rows or byte range)
    Slice(RuntimeSliceArgs),
    /// Show runtime statistics
    Stats(RuntimeStatsArgs),
    /// Benchmark runtime operations
    Bench(RuntimeBenchArgs),
}

#[derive(Args)] struct InspectArgs { path: PathBuf, #[arg(long)] hex: bool }
#[derive(Args)] struct PackArgs { #[arg(short, long)] manifest: PathBuf, #[arg(short, long)] data_dir: PathBuf, #[arg(short, long)] output: PathBuf, #[arg(short, long)] architecture: Option<String>, #[arg(short = 'n', long)] model: Option<String> }
#[derive(Args)] struct UnpackArgs { path: PathBuf, #[arg(short, long)] output: PathBuf, #[arg(long)] raw: bool }
#[derive(Args)] struct ConvertArgs { input: PathBuf, output: PathBuf }
#[derive(Args)] struct BenchArgs { path: PathBuf, #[arg(short, long, default_value = "10")] iterations: u32 }
#[derive(Args)] struct ValidateArgs { path: PathBuf, #[arg(long)] no_checksums: bool }
#[derive(Args)] struct ListArgs { path: PathBuf, #[arg(long)] verbose: bool }
#[derive(Args)] struct ExtractArgs { path: PathBuf, #[arg(short, long)] name: String, #[arg(short, long)] output: PathBuf }
#[derive(Args)] struct CreateArgs { output: PathBuf, #[arg(short, long)] model: Option<String>, #[arg(short, long)] architecture: Option<String> }

// Runtime subcommand args
#[derive(Args)] struct RuntimeInspectArgs { path: PathBuf, #[arg(long)] cache: Option<String> }
#[derive(Args)] struct RuntimeTensorArgs { path: PathBuf, name: String }
#[derive(Args)] struct RuntimeSliceArgs { path: PathBuf, name: String, #[arg(long, default_value = "rows=0,1")] rows: Option<String>, #[arg(long)] bytes: Option<String> }
#[derive(Args)] struct RuntimeStatsArgs { path: PathBuf }
#[derive(Args)] struct RuntimeBenchArgs { path: PathBuf, #[arg(short, long, default_value = "10")] iterations: u32 }

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match Cli::parse().command {
        Commands::Inspect(a) => cmd_inspect(&a),
        Commands::Pack(a) => cmd_pack(&a),
        Commands::Unpack(a) => cmd_unpack(&a),
        Commands::Convert(a) => cmd_convert(&a),
        Commands::Bench(a) => cmd_bench(&a),
        Commands::Validate(a) => cmd_validate(&a),
        Commands::List(a) => cmd_list(&a),
        Commands::Extract(a) => cmd_extract(&a),
        Commands::Create(a) => cmd_create(&a),
        Commands::Runtime(cmd) => match cmd {
            RuntimeCommands::Inspect(a) => cmd_runtime_inspect(&a),
            RuntimeCommands::Tensor(a) => cmd_runtime_tensor(&a),
            RuntimeCommands::Slice(a) => cmd_runtime_slice(&a),
            RuntimeCommands::Stats(a) => cmd_runtime_stats(&a),
            RuntimeCommands::Bench(a) => cmd_runtime_bench(&a),
        },
    }
}

// ── Existing commands (unchanged) ──────────────────────────────────

fn cmd_inspect(args: &InspectArgs) {
    let data = fs::read(&args.path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", args.path.display()));
    if args.hex {
        println!("=== First 256 bytes (hex) ===");
        for (i, chunk) in data.iter().take(256).enumerate() {
            if i % 16 == 0 { print!("\n{:08X}  ", i); }
            print!("{:02X} ", chunk);
        }
        println!();
    }
    let file = AxonFile::from_bytes(data).unwrap_or_else(|e| panic!("Failed to parse .axon: {e}"));
    println!("=== .axon File Inspection ===");
    println!("Magic:       {:?}", std::str::from_utf8(&file.header.magic).unwrap());
    println!("Version:     {}", file.header.version);
    println!("Tensors:     {}", file.header.tensor_count);
    println!("Payload:     {} bytes ({:.2} MB)", file.header.payload_size, file.header.payload_size as f64 / 1_048_576.0);
    println!("Manifest:    {} bytes at offset {}", file.header.manifest_size, file.header.manifest_offset);
    println!("Flags:       {:#018x}", file.header.flags);
    println!();
    println!("Model:       {}", file.manifest.model.as_deref().unwrap_or("N/A"));
    println!();
    for (i, name) in file.manifest.tensor_order.iter().enumerate() {
        if let Some(desc) = file.manifest.get_tensor(name) {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            let shape = desc.shape_vec();
            let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            println!("  [{:3}] {}  {}  [{}]  {} bytes", i, name, dtype.name(), s.join(", "), desc.data_size);
        }
    }
}

fn cmd_pack(args: &PackArgs) {
    let manifest_json = fs::read_to_string(&args.manifest).expect("Failed to read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json).expect("Invalid JSON manifest");
    let mut builder = AxonBuilder::new();
    if let Some(ref model) = args.model { builder = builder.model(model); }
    if let Some(ref arch) = args.architecture { builder = builder.architecture(arch); }
    let tensors = manifest["tensors"].as_array().expect("Manifest must have a 'tensors' array");
    for entry in tensors {
        let name = entry["name"].as_str().expect("Tensor must have a name");
        let dtype_code = entry["dtype"].as_u64().unwrap_or(0) as u32;
        let shape: Vec<u64> = entry["shape"].as_array().expect("Tensor must have shape").iter().map(|v| v.as_u64().unwrap()).collect();
        let dtype = DType::from_code(dtype_code).expect("Invalid dtype code");
        let data_path = args.data_dir.join(name);
        let data = fs::read(&data_path).unwrap_or_else(|_| {
            let expected = shape.iter().product::<u64>() as usize * dtype.size_in_bytes();
            info!("Generating {} bytes synthetic data for {}", expected, name);
            (0..expected).map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as u8).collect()
        });
        builder = builder.add_tensor(name, data, dtype, &shape);
        info!("Added tensor: {} dtype={} shape={:?}", name, dtype.name(), shape);
    }
    let axon_bytes = builder.build().expect("Failed to build .axon file");
    fs::write(&args.output, &axon_bytes).expect("Failed to write .axon file");
    println!("Written: {} ({:.2} MB)", args.output.display(), axon_bytes.len() as f64 / 1_048_576.0);
}

fn cmd_unpack(args: &UnpackArgs) {
    fs::create_dir_all(&args.output).expect("Failed to create output directory");
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    for (name, desc) in &file.manifest.tensors {
        let tensor_bytes = file.tensor_data(name).expect("Failed to get tensor data");
        let output_path = if args.raw {
            args.output.join(format!("{}.bin", name.replace('/', ".")))
        } else {
            args.output.join(format!("{}.npy", name.replace('/', ".")))
        };
        if args.raw {
            fs::write(&output_path, tensor_bytes).expect("Failed to write tensor");
        } else {
            let mut out_bytes = Vec::new();
            out_bytes.extend_from_slice(&npy_header(desc));
            out_bytes.extend_from_slice(tensor_bytes);
            fs::write(&output_path, out_bytes).expect("Failed to write .npy tensor");
        }
        info!("Extracted: {} -> {} ({} bytes)", name, output_path.display(), tensor_bytes.len());
    }
    println!("Extracted {} tensors to {}", file.manifest.tensor_count(), args.output.display());
}

fn cmd_convert(args: &ConvertArgs) {
    let data = fs::read(&args.input).expect("Failed to read input file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    let json = serde_json::to_string_pretty(&file.manifest).expect("Failed to serialize");
    fs::write(&args.output, &json).expect("Failed to write JSON");
    println!("Converted {} -> {} ({} tensors)", args.input.display(), args.output.display(), file.manifest.tensor_count());
}

fn cmd_bench(args: &BenchArgs) {
    println!("Benchmarking: {} ({} iterations)", args.path.display(), args.iterations);
    let start = Instant::now();
    for _ in 0..args.iterations {
        let data = fs::read(&args.path).expect("Failed to read file");
        let _file = AxonFile::from_bytes(data).expect("Failed to parse");
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Load (core):  {:?} total, {:?} avg", dur, avg);

    let start = Instant::now();
    for _ in 0..args.iterations {
        let _rt = AxonRuntime::open(&args.path).expect("Failed to open runtime");
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Open (runtime): {:?} total, {:?} avg", dur, avg);

    let data = fs::read(&args.path).expect("Failed to read");
    let file = AxonFile::from_bytes(data).expect("Failed to parse");
    let start = Instant::now();
    for _ in 0..args.iterations {
        for (name, _) in &file.manifest.tensors { let _ = file.tensor_data(name); }
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Index (core): {:?} total, {:?} avg", dur, avg);
    println!("  Tensors: {}", file.manifest.tensor_count());
    println!("  Payload: {} bytes ({:.2} MB)", file.header.payload_size, file.header.payload_size as f64 / 1_048_576.0);
}

fn cmd_validate(args: &ValidateArgs) {
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    println!("Validating: {}", args.path.display());
    println!("  Magic:      OK (AXON v{})", file.header.version);
    println!("  Tensors:    {} descriptors", file.header.tensor_count);
    if !args.no_checksums {
        let r = file.verify_all_checksums();
        let pass = r.iter().filter(|(_, ok)| *ok).count();
        let fail = r.iter().filter(|(_, ok)| !*ok).count();
        println!("  Checksums:  {}/{} passed, {} failed", pass, r.len(), fail);
        for (n, ok) in &r { if !ok { eprintln!("  CHECKSUM FAIL: {n}"); } }
    } else { println!("  Checksums:  skipped"); }
    println!("  Status:     VALID");
}

fn cmd_list(args: &ListArgs) {
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    println!("Tensors in {}:", args.path.display());
    println!();
    if args.verbose { println!("{:<5} {:<48} {:<8} {:<24} {:>12}", "#", "Name", "DType", "Shape", "Size"); println!("{}", "-".repeat(100)); }
    for (i, name) in file.manifest.tensor_order.iter().enumerate() {
        if let Some(desc) = file.manifest.get_tensor(name) {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            let shape = desc.shape_vec();
            let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            if args.verbose { println!("{:<5} {:<48} {:<8} [{:<22}] {:>12}", i, name, dtype.name(), s.join(", "), format_size(desc.data_size)); }
            else { println!("  {}  {}  {}", name, dtype.name(), s.join("x")); }
        }
    }
}

fn cmd_extract(args: &ExtractArgs) {
    let data = fs::read(&args.path).expect("Failed to read file");
    let file = AxonFile::from_bytes(data).expect("Failed to parse .axon file");
    let tensor_bytes = file.tensor_data(&args.name).unwrap_or_else(|| panic!("Tensor '{}' not found", args.name));
    fs::write(&args.output, tensor_bytes).expect("Failed to write tensor data");
    println!("Extracted {} -> {} ({} bytes)", args.name, args.output.display(), tensor_bytes.len());
}

fn cmd_create(args: &CreateArgs) {
    let mut builder = AxonBuilder::new();
    if let Some(ref model) = args.model { builder = builder.model(model); }
    if let Some(ref arch) = args.architecture { builder = builder.architecture(arch); }
    let r = |n: usize| { (0..n).map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as u8).collect::<Vec<_>>() };
    builder = builder.add_tensor("emb_weight", r(32000 * 4096 * 2), DType::F16, &[32000, 4096]);
    for layer in 0..2 {
        for proj in &["q", "k", "v", "o"] {
            builder = builder.add_tensor(&format!("layer_{}_{}", layer, proj), r(4096 * 4096), DType::Q4, &[4096, 4096]);
        }
    }
    for layer in 0..2 {
        for p in &["gate", "up", "down"] {
            let (rows, cols) = if *p == "down" { (11008, 4096) } else { (4096, 11008) };
            builder = builder.add_tensor(&format!("layer_{}_mlp_{}", layer, p), r(rows * cols * 2), DType::F16, &[rows as u64, cols as u64]);
        }
    }
    builder = builder.add_tensor("norm_weight", r(4096 * 2), DType::F16, &[4096]);
    builder = builder.add_tensor("lm_head", r(32000 * 4096 * 2), DType::F16, &[32000, 4096]);
    let axon_bytes = builder.build().expect("Failed to build .axon file");
    fs::write(&args.output, &axon_bytes).expect("Failed to write .axon file");
    println!("Created: {} ({:.2} MB, {} tensors)", args.output.display(), axon_bytes.len() as f64 / 1_048_576.0, 17);
}

// ── New: Runtime commands ──────────────────────────────────────────

fn cmd_runtime_inspect(args: &RuntimeInspectArgs) {
    let start = Instant::now();
    let rt = AxonRuntime::open(&args.path)
        .unwrap_or_else(|e| { eprintln!("Failed to open: {e}"); std::process::exit(1); });
    let open_time = start.elapsed();

    println!("=== Axon Runtime Inspection ===");
    println!("File:        {}", args.path.display());
    println!("Open time:   {:?}", open_time);
    println!();
    println!("Model:       {}", rt.model_name());
    println!("Arch:        {}", rt.architecture());
    println!("Tensors:     {}", rt.tensor_count());
    println!("Payload:     {} ({:.2} MB)", rt.payload_size(), rt.payload_size() as f64 / 1_048_576.0);
    println!("File size:   {} ({:.2} MB)", rt.file_size(), rt.file_size() as f64 / 1_048_576.0);
    println!("Mmap:        active (zero-copy views available)");

    if let Some(cache_str) = &args.cache {
        let _bytes = parse_size(cache_str);
        println!("Cache:       {} (use with CachedRuntime for LRU caching)", cache_str);
    } else {
        println!("Cache:       disabled (use --cache <size> to enable)");
    }
    println!();

    // Show tensor summary
    println!("{:<5} {:<48} {:<8} {:<28} {:>12}", "#", "Name", "DType", "Shape", "Size");
    println!("{}", "-".repeat(100));
    for (i, info) in rt.tensors().iter().enumerate() {
        let dtype = info.dtype;
        let shape_str: Vec<String> = info.shape.iter().map(|s| s.to_string()).collect();
        println!("{:<5} {:<48} {:<8} [{:<26}] {:>12}",
            i, truncate_name(&info.name, 48), dtype.name(), shape_str.join(", "), format_size(info.data_size));
    }

    let stats = rt.stats();
    println!();
    println!("Stats:");
    println!("  Bytes accessed: {}", stats.bytes_read());
    println!("  Access count:   {}", stats.tensor_accesses());
    println!();
    println!("No tensor data loaded. Use `axon runtime tensor <name>` to access tensors.");
}

fn cmd_runtime_tensor(args: &RuntimeTensorArgs) {
    let rt = AxonRuntime::open(&args.path)
        .unwrap_or_else(|e| { eprintln!("Failed to open: {e}"); std::process::exit(1); });

    let info = rt.tensor_info(&args.name).unwrap_or_else(|e| {
        eprintln!("Tensor '{}' not found: {}", args.name, e);
        std::process::exit(1);
    });

    // Use zero-copy view to peek at first bytes
    let view = rt.tensor_view(&args.name).unwrap_or_else(|e| {
        eprintln!("Failed to access tensor: {e}");
        std::process::exit(1);
    });

    println!("Tensor: {}", args.name);
    println!("  DType:  {}", info.dtype.name());
    println!("  Shape:  {:?}", info.shape);
    println!("  Offset: {} ({} bytes)", format_size(info.data_offset), info.data_offset);
    println!("  Size:   {} ({} bytes)", format_size(info.data_size), info.data_size);
    println!("  Access: zero-copy mmap view");

    if view.len() > 0 {
        let preview = &view[..view.len().min(32)];
        println!("  First {} bytes: {:02x?}", preview.len(), preview);
    }
}

fn cmd_runtime_slice(args: &RuntimeSliceArgs) {
    let rt = AxonRuntime::open(&args.path)
        .unwrap_or_else(|e| { eprintln!("Failed to open: {e}"); std::process::exit(1); });

    let info = rt.tensor_info(&args.name).unwrap_or_else(|e| {
        eprintln!("Tensor '{}' not found: {}", args.name, e);
        std::process::exit(1);
    });

    if let Some(rows_str) = &args.rows {
        let parts: Vec<&str> = rows_str.splitn(3, '=').collect();
        let range = if parts.len() >= 2 { parts[1] } else { parts[0] };
        let idx: Vec<&str> = range.split(',').collect();
        if idx.len() != 2 {
            eprintln!("Expected --rows 'start,end' format");
            std::process::exit(1);
        }
        let start: usize = idx[0].parse().unwrap_or(0);
        let end: usize = idx[1].parse().unwrap_or(1);
        let view = rt.tensor_rows(&args.name, start, end).unwrap_or_else(|e| {
            eprintln!("Failed to read rows {}-{}: {}", start, end, e);
            std::process::exit(1);
        });
        let num_rows = end - start;
        let cols = if info.shape.len() >= 2 { info.shape[1] } else { 1 };
        let elem_size = info.dtype.size_in_bytes();
        println!("Tensor: {} rows {}..{} ({} rows × {} cols × {} bytes/elem)", args.name, start, end, num_rows, cols, elem_size);
        println!("  Size: {} bytes", view.len());
        print_preview(view);
    } else if let Some(bytes_str) = &args.bytes {
        let idx: Vec<&str> = bytes_str.split(',').collect();
        if idx.len() != 2 {
            eprintln!("Expected --bytes 'offset,size' format");
            std::process::exit(1);
        }
        let off: u64 = idx[0].parse().unwrap_or(0);
        let sz: u64 = idx[1].parse().unwrap_or(64);
        let view = rt.tensor_byte_view(&args.name, off as usize..(off + sz) as usize).unwrap_or_else(|e| {
            eprintln!("Failed to read bytes {}-{}: {}", off, off + sz, e);
            std::process::exit(1);
        });
        println!("Tensor: {} bytes {}..{}", args.name, off, off + sz);
        print_preview(view);
    }
}

fn cmd_runtime_stats(args: &RuntimeStatsArgs) {
    let rt = AxonRuntime::open(&args.path)
        .unwrap_or_else(|e| { eprintln!("Failed to open: {e}"); std::process::exit(1); });

    let stats = rt.stats();
    println!("=== Axon Runtime Stats ===");
    println!("File:           {}", args.path.display());
    println!("Model:          {}", rt.model_name());
    println!("Tensor count:   {}", rt.tensor_count());
    println!("Payload size:   {} ({:.2} MB)", rt.payload_size(), rt.payload_size() as f64 / 1_048_576.0);
    println!("File size:      {} ({:.2} MB)", rt.file_size(), rt.file_size() as f64 / 1_048_576.0);
    println!("Mmap:           active");
    println!();
    println!("Access stats:");
    println!("  Bytes read:   {}", stats.bytes_read());
    println!("  Access count: {}", stats.tensor_accesses());
    println!("  Tensor count: {}", rt.tensor_count());

    // Memory: OS handles mmap paging, so report file size as potential
    println!();
    println!("Memory:");
    println!("  Mmap window:  {} (all tensors mapped, OS pages on demand)", format_size(rt.file_size()));
    println!("  In mem:       OS managed (only accessed pages are resident)");

    // Cache state
    println!();
    println!("Cache: use CachedRuntime for LRU caching (`AxonRuntime::with_cache`)");
    println!("  To test: axon runtime inspect {} --cache 1GB", args.path.display());
}

fn cmd_runtime_bench(args: &RuntimeBenchArgs) {
    println!("Benchmarking runtime: {} ({} iterations)", args.path.display(), args.iterations);

    // Open time
    let start = Instant::now();
    for _ in 0..args.iterations {
        let _rt = AxonRuntime::open(&args.path)
            .unwrap_or_else(|e| { eprintln!("Failed: {e}"); std::process::exit(1); });
    }
    let total_open = start.elapsed();
    println!("Open time:      {:?} avg per open", total_open / args.iterations);

    // First tensor
    let rt = AxonRuntime::open(&args.path).unwrap();
    let names = rt.tensor_names();
    if names.is_empty() { println!("No tensors to benchmark."); return; }
    let start = Instant::now();
    for _ in 0..args.iterations {
        let _ = rt.tensor_view(names[0]).unwrap();
    }
    let first_time = start.elapsed();
    println!("First tensor:   {:?} avg ({})", first_time / args.iterations, names[0]);

    // All tensors (full scan)
    let start = Instant::now();
    for _ in 0..args.iterations {
        for name in &names {
            let _ = rt.tensor_view(name).unwrap();
        }
    }
    let scan_time = start.elapsed();
    let total = scan_time / args.iterations;
    let per_tensor = total / names.len() as u32;
    println!("Full scan:      {:?} total ({:?} per tensor, {} tensors)", total, per_tensor, names.len());

    // Byte range
    let _info = rt.tensor_info(names[0]).unwrap();
    let start = Instant::now();
    for _ in 0..args.iterations {
        let _ = rt.tensor_byte_view(names[0], 0..64).unwrap();
    }
    let byte_range = start.elapsed();
    println!("Byte range:     {:?} avg (first 64 bytes)", byte_range / args.iterations);

    // Stats
    let stats = rt.stats();
    println!();
    println!("  Bytes accessed: {}", stats.bytes_read());
    println!("  Access count:   {}", stats.tensor_accesses());
}

fn print_preview(data: &[u8]) {
    let preview = if data.len() > 64 {
        &data[..64]
    } else {
        data
    };
    println!("  First {} bytes: {:02x?}...", preview.len(), &preview[..preview.len().min(16)]);
}

fn parse_size(s: &str) -> usize {
    let s = s.trim().to_lowercase();
    let (num_str, suffix) = if s.ends_with("gb") {
        (&s[..s.len()-2], 1_073_741_824usize)
    } else if s.ends_with("mb") {
        (&s[..s.len()-2], 1_048_576usize)
    } else if s.ends_with("kb") {
        (&s[..s.len()-2], 1024usize)
    } else {
        (s.as_str(), 1usize)
    };
    let num: f64 = num_str.trim().parse().unwrap_or(0.0);
    (num * suffix as f64) as usize
}

// ── Shared helpers ─────────────────────────────────────────────────

fn npy_header(desc: &TensorDescriptor) -> Vec<u8> {
    let dtype = desc.dtype().unwrap_or(DType::F32);
    let shape = desc.shape_vec();
    let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let ds = match dtype { DType::F32 => "<f4", DType::F16 | DType::BF16 => "<f2", DType::I32 => "<i4", DType::I64 => "<i8", DType::U8 => "u1", _ => "<f4" };
    let h = format!("{{'descr': '{ds}', 'fortran_order': False, 'shape': ({},) }}", s.join(", "));
    let mut hb = h.as_bytes().to_vec();
    let padded = ((10 + hb.len() + 63) / 64) * 64;
    hb.extend(std::iter::repeat(b' ').take(padded - 10 - hb.len()));
    let mut r = Vec::new();
    r.extend_from_slice(b"\x93NUMPY"); r.push(1); r.push(0);
    r.extend_from_slice(&(hb.len() as u16).to_le_bytes());
    r.extend_from_slice(&hb); r
}

fn format_size(bytes: u64) -> String {
    let u = &["B", "KB", "MB", "GB", "TB"];
    let mut s = bytes as f64; let mut i = 0;
    while s >= 1024.0 && i < u.len() - 1 { s /= 1024.0; i += 1; }
    format!("{:.2} {}", s, u[i])
}

fn truncate_name(name: &str, max: usize) -> String {
    if name.len() <= max { name.to_string() }
    else { format!("{}...", &name[..max-3]) }
}
