use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use axon_core::*;
use axon_runtime::AxonRuntime;
use clap::{Args, Parser, Subcommand};
use log::info;

type CliResult = Result<(), String>;

#[derive(Parser)]
#[command(
    name = "axon",
    about = "Adaptive eXecutable Object Notation CLI",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Inspect(InspectArgs),
    Pack(PackArgs),
    Unpack(UnpackArgs),
    Convert(ConvertArgs),
    Bench(BenchArgs),
    Validate(ValidateArgs),
    List(ListArgs),
    Extract(ExtractArgs),
    Create(CreateArgs),
    #[command(subcommand)]
    Runtime(RuntimeCommands),
}

#[derive(Subcommand)]
enum RuntimeCommands {
    /// Show detailed runtime information about a model file
    Inspect(RuntimeInspectArgs),
    /// Access a tensor and print its size and first bytes
    Tensor(RuntimeTensorArgs),
    /// Slice a tensor by rows or byte range
    Slice(RuntimeSliceArgs),
    /// Show runtime statistics
    Stats(RuntimeStatsArgs),
    /// Benchmark runtime operations
    Bench(RuntimeBenchArgs),
}

#[derive(Args)]
struct InspectArgs {
    path: PathBuf,
    #[arg(long)]
    hex: bool,
}

#[derive(Args)]
struct PackArgs {
    #[arg(short, long)]
    manifest: PathBuf,
    #[arg(short, long)]
    data_dir: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long)]
    architecture: Option<String>,
    #[arg(short = 'n', long)]
    model: Option<String>,
}

#[derive(Args)]
struct UnpackArgs {
    path: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    raw: bool,
}

#[derive(Args)]
struct ConvertArgs {
    input: PathBuf,
    output: PathBuf,
}

#[derive(Args)]
struct BenchArgs {
    path: PathBuf,
    #[arg(short, long, default_value = "10")]
    iterations: u32,
}

#[derive(Args)]
struct ValidateArgs {
    path: PathBuf,
    #[arg(long)]
    no_checksums: bool,
}

#[derive(Args)]
struct ListArgs {
    path: PathBuf,
    #[arg(long)]
    verbose: bool,
}

#[derive(Args)]
struct ExtractArgs {
    path: PathBuf,
    #[arg(short, long)]
    name: String,
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct CreateArgs {
    output: PathBuf,
    #[arg(short, long)]
    model: Option<String>,
    #[arg(short, long)]
    architecture: Option<String>,
}

#[derive(Args)]
struct RuntimeInspectArgs {
    path: PathBuf,
    #[arg(long)]
    cache: Option<String>,
}

#[derive(Args)]
struct RuntimeTensorArgs {
    path: PathBuf,
    name: String,
}

#[derive(Args)]
struct RuntimeSliceArgs {
    path: PathBuf,
    name: String,
    #[arg(long)]
    rows: Option<String>,
    #[arg(long)]
    bytes: Option<String>,
}

#[derive(Args)]
struct RuntimeStatsArgs {
    path: PathBuf,
}

#[derive(Args)]
struct RuntimeBenchArgs {
    path: PathBuf,
    #[arg(short, long, default_value = "10")]
    iterations: u32,
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let result = match Cli::parse().command {
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
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_inspect(args: &InspectArgs) -> CliResult {
    let data =
        fs::read(&args.path).map_err(|e| format!("failed to read {}: {e}", args.path.display()))?;
    if args.hex {
        println!("=== First 256 bytes (hex) ===");
        for (i, chunk) in data.iter().take(256).enumerate() {
            if i % 16 == 0 {
                print!("\n{:08X}  ", i);
            }
            print!("{:02X} ", chunk);
        }
        println!();
    }

    let file = AxonFile::from_bytes(data).map_err(|e| format!("failed to parse .axon: {e}"))?;
    println!("=== .axon File Inspection ===");
    println!(
        "Magic:       {:?}",
        String::from_utf8_lossy(&file.header.magic)
    );
    println!("Version:     {}", file.header.version);
    println!("Tensors:     {}", file.header.tensor_count);
    println!(
        "Payload:     {} bytes ({:.2} MB)",
        file.header.payload_size,
        file.header.payload_size as f64 / 1_048_576.0
    );
    println!(
        "Manifest:    {} bytes at offset {}",
        file.header.manifest_size, file.header.manifest_offset
    );
    println!("Flags:       {:#018x}", file.header.flags);
    println!();
    println!(
        "Model:       {}",
        file.manifest.model.as_deref().unwrap_or("N/A")
    );
    println!();

    for (i, name) in file.manifest.tensor_order.iter().enumerate() {
        if let Some(desc) = file.manifest.get_tensor(name) {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            let shape = desc.shape_vec();
            let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            println!(
                "  [{:3}] {}  {}  [{}]  {} bytes",
                i,
                name,
                dtype.name(),
                s.join(", "),
                desc.data_size
            );
        }
    }
    Ok(())
}

fn cmd_pack(args: &PackArgs) -> CliResult {
    let manifest_json = fs::read_to_string(&args.manifest)
        .map_err(|e| format!("failed to read manifest {}: {e}", args.manifest.display()))?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_json).map_err(|e| format!("invalid JSON manifest: {e}"))?;

    let mut builder = AxonBuilder::new();
    if let Some(ref model) = args.model {
        builder = builder.model(model);
    }
    if let Some(ref arch) = args.architecture {
        builder = builder.architecture(arch);
    }

    let tensors = manifest
        .get("tensors")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "manifest must contain a 'tensors' array".to_string())?;

    for (idx, entry) in tensors.iter().enumerate() {
        let name = entry
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| format!("tensor #{idx} is missing a string 'name'"))?;
        let dtype_code = entry
            .get("dtype")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as u32;
        let shape_values = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("tensor {name} is missing a 'shape' array"))?;
        let shape: Vec<u64> = shape_values
            .iter()
            .enumerate()
            .map(|(dim_idx, value)| {
                value
                    .as_u64()
                    .ok_or_else(|| format!("tensor {name} shape[{dim_idx}] must be an integer"))
            })
            .collect::<Result<_, _>>()?;
        let dtype = DType::from_code(dtype_code)
            .map_err(|e| format!("tensor {name} has invalid dtype {dtype_code}: {e}"))?;
        let data_path = args.data_dir.join(name);
        let data = fs::read(&data_path).unwrap_or_else(|_| {
            let expected = shape.iter().product::<u64>() as usize * dtype.size_in_bytes();
            info!("Generating {} bytes synthetic data for {}", expected, name);
            pseudo_random_bytes(expected)
        });
        builder = builder.add_tensor(name, data, dtype, &shape);
        info!(
            "Added tensor: {} dtype={} shape={:?}",
            name,
            dtype.name(),
            shape
        );
    }

    let axon_bytes = builder
        .build()
        .map_err(|e| format!("failed to build .axon file: {e}"))?;
    fs::write(&args.output, &axon_bytes)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
    println!(
        "Written: {} ({:.2} MB)",
        args.output.display(),
        axon_bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}

fn cmd_unpack(args: &UnpackArgs) -> CliResult {
    fs::create_dir_all(&args.output).map_err(|e| {
        format!(
            "failed to create output directory {}: {e}",
            args.output.display()
        )
    })?;
    let file = read_axon(&args.path)?;

    for (name, desc) in &file.manifest.tensors {
        let tensor_bytes = file
            .tensor_data(name)
            .ok_or_else(|| format!("tensor {name} data is not available"))?;
        let output_path = if args.raw {
            args.output.join(format!("{}.bin", name.replace('/', ".")))
        } else {
            args.output.join(format!("{}.npy", name.replace('/', ".")))
        };
        if args.raw {
            fs::write(&output_path, tensor_bytes)
                .map_err(|e| format!("failed to write {}: {e}", output_path.display()))?;
        } else {
            let mut out_bytes = Vec::new();
            out_bytes.extend_from_slice(&npy_header(desc));
            out_bytes.extend_from_slice(tensor_bytes);
            fs::write(&output_path, out_bytes)
                .map_err(|e| format!("failed to write {}: {e}", output_path.display()))?;
        }
        info!(
            "Extracted: {} -> {} ({} bytes)",
            name,
            output_path.display(),
            tensor_bytes.len()
        );
    }
    println!(
        "Extracted {} tensors to {}",
        file.manifest.tensor_count(),
        args.output.display()
    );
    Ok(())
}

fn cmd_convert(args: &ConvertArgs) -> CliResult {
    let file = read_axon(&args.input)?;
    let json = serde_json::to_string_pretty(&file.manifest)
        .map_err(|e| format!("failed to serialize manifest: {e}"))?;
    fs::write(&args.output, &json)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
    println!(
        "Converted {} -> {} ({} tensors)",
        args.input.display(),
        args.output.display(),
        file.manifest.tensor_count()
    );
    Ok(())
}

fn cmd_bench(args: &BenchArgs) -> CliResult {
    validate_iterations(args.iterations)?;
    println!(
        "Benchmarking: {} ({} iterations)",
        args.path.display(),
        args.iterations
    );

    let start = Instant::now();
    for _ in 0..args.iterations {
        let _file = read_axon(&args.path)?;
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Load (core):  {:?} total, {:?} avg", dur, avg);

    let start = Instant::now();
    for _ in 0..args.iterations {
        let _rt = open_runtime(&args.path)?;
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Open (runtime): {:?} total, {:?} avg", dur, avg);

    let file = read_axon(&args.path)?;
    let start = Instant::now();
    for _ in 0..args.iterations {
        for name in file.manifest.tensors.keys() {
            let _ = file.tensor_data(name);
        }
    }
    let dur = start.elapsed();
    let avg = dur / args.iterations;
    println!("  Index (core): {:?} total, {:?} avg", dur, avg);
    println!("  Tensors: {}", file.manifest.tensor_count());
    println!(
        "  Payload: {} bytes ({:.2} MB)",
        file.header.payload_size,
        file.header.payload_size as f64 / 1_048_576.0
    );
    Ok(())
}

fn cmd_validate(args: &ValidateArgs) -> CliResult {
    let file = read_axon(&args.path)?;
    println!("Validating: {}", args.path.display());
    println!("  Magic:      OK (AXON v{})", file.header.version);
    println!("  Tensors:    {} descriptors", file.header.tensor_count);
    if !args.no_checksums {
        let results = file.verify_all_checksums();
        let pass = results.iter().filter(|(_, ok)| *ok).count();
        let fail = results.iter().filter(|(_, ok)| !*ok).count();
        println!(
            "  Checksums:  {}/{} passed, {} failed",
            pass,
            results.len(),
            fail
        );
        for (name, ok) in &results {
            if !ok {
                eprintln!("  CHECKSUM FAIL: {name}");
            }
        }
        if fail > 0 {
            return Err(format!("{fail} checksum(s) failed"));
        }
    } else {
        println!("  Checksums:  skipped");
    }
    println!("  Status:     VALID");
    Ok(())
}

fn cmd_list(args: &ListArgs) -> CliResult {
    let file = read_axon(&args.path)?;
    println!("Tensors in {}:", args.path.display());
    println!();
    if args.verbose {
        println!(
            "{:<5} {:<48} {:<8} {:<24} {:>12}",
            "#", "Name", "DType", "Shape", "Size"
        );
        println!("{}", "-".repeat(100));
    }
    for (i, name) in file.manifest.tensor_order.iter().enumerate() {
        if let Some(desc) = file.manifest.get_tensor(name) {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            let shape = desc.shape_vec();
            let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            if args.verbose {
                println!(
                    "{:<5} {:<48} {:<8} [{:<22}] {:>12}",
                    i,
                    name,
                    dtype.name(),
                    s.join(", "),
                    format_size(desc.data_size)
                );
            } else {
                println!("  {}  {}  {}", name, dtype.name(), s.join("x"));
            }
        }
    }
    Ok(())
}

fn cmd_extract(args: &ExtractArgs) -> CliResult {
    let file = read_axon(&args.path)?;
    let tensor_bytes = file
        .tensor_data(&args.name)
        .ok_or_else(|| format!("tensor '{}' not found", args.name))?;
    fs::write(&args.output, tensor_bytes)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
    println!(
        "Extracted {} -> {} ({} bytes)",
        args.name,
        args.output.display(),
        tensor_bytes.len()
    );
    Ok(())
}

fn cmd_create(args: &CreateArgs) -> CliResult {
    let mut builder = AxonBuilder::new();
    if let Some(ref model) = args.model {
        builder = builder.model(model);
    }
    if let Some(ref arch) = args.architecture {
        builder = builder.architecture(arch);
    }

    builder = builder.add_tensor(
        "emb_weight",
        pseudo_random_bytes(32000 * 4096 * 2),
        DType::F16,
        &[32000, 4096],
    );
    for layer in 0..2 {
        for proj in &["q", "k", "v", "o"] {
            builder = builder.add_tensor(
                &format!("layer_{}_{}", layer, proj),
                pseudo_random_bytes(4096 * 4096),
                DType::Q4,
                &[4096, 4096],
            );
        }
    }
    for layer in 0..2 {
        for p in &["gate", "up", "down"] {
            let (rows, cols) = if *p == "down" {
                (11008, 4096)
            } else {
                (4096, 11008)
            };
            builder = builder.add_tensor(
                &format!("layer_{}_mlp_{}", layer, p),
                pseudo_random_bytes(rows * cols * 2),
                DType::F16,
                &[rows as u64, cols as u64],
            );
        }
    }
    builder = builder.add_tensor(
        "norm_weight",
        pseudo_random_bytes(4096 * 2),
        DType::F16,
        &[4096],
    );
    builder = builder.add_tensor(
        "lm_head",
        pseudo_random_bytes(32000 * 4096 * 2),
        DType::F16,
        &[32000, 4096],
    );
    let axon_bytes = builder
        .build()
        .map_err(|e| format!("failed to build .axon file: {e}"))?;
    fs::write(&args.output, &axon_bytes)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
    println!(
        "Created: {} ({:.2} MB, {} tensors)",
        args.output.display(),
        axon_bytes.len() as f64 / 1_048_576.0,
        17
    );
    Ok(())
}

fn cmd_runtime_inspect(args: &RuntimeInspectArgs) -> CliResult {
    let start = Instant::now();
    let rt = open_runtime(&args.path)?;
    let open_time = start.elapsed();

    println!("=== Axon Runtime Inspection ===");
    println!("File:        {}", args.path.display());
    println!("Open time:   {:?}", open_time);
    println!();
    println!("Model:       {}", rt.model_name());
    println!("Arch:        {}", rt.architecture());
    println!("Tensors:     {}", rt.tensor_count());
    println!(
        "Payload:     {} ({:.2} MB)",
        rt.payload_size(),
        rt.payload_size() as f64 / 1_048_576.0
    );
    println!(
        "File size:   {} ({:.2} MB)",
        rt.file_size(),
        rt.file_size() as f64 / 1_048_576.0
    );
    println!("Mmap:        active (zero-copy views available)");

    if let Some(cache_str) = &args.cache {
        let _bytes = parse_size(cache_str);
        println!(
            "Cache:       {} (use with CachedRuntime for LRU caching)",
            cache_str
        );
    } else {
        println!("Cache:       disabled (use --cache <size> to enable)");
    }
    println!();

    println!(
        "{:<5} {:<48} {:<8} {:<28} {:>12}",
        "#", "Name", "DType", "Shape", "Size"
    );
    println!("{}", "-".repeat(100));
    for (i, info) in rt.tensors().iter().enumerate() {
        let shape_str: Vec<String> = info.shape.iter().map(|s| s.to_string()).collect();
        println!(
            "{:<5} {:<48} {:<8} [{:<26}] {:>12}",
            i,
            truncate_name(&info.name, 48),
            info.dtype.name(),
            shape_str.join(", "),
            format_size(info.data_size)
        );
    }

    let stats = rt.stats();
    println!();
    println!("Stats:");
    println!("  Bytes accessed: {}", stats.bytes_read());
    println!("  Access count:   {}", stats.tensor_accesses());
    println!();
    println!("No tensor data loaded. Use `axon runtime tensor <file> <name>` to access tensors.");
    Ok(())
}

fn cmd_runtime_tensor(args: &RuntimeTensorArgs) -> CliResult {
    let rt = open_runtime(&args.path)?;
    let info = rt
        .tensor_info(&args.name)
        .map_err(|e| format!("tensor '{}' not found: {e}", args.name))?;
    let view = rt
        .tensor_view(&args.name)
        .map_err(|e| format!("failed to access tensor '{}': {e}", args.name))?;

    println!("Tensor: {}", args.name);
    println!("  DType:  {}", info.dtype.name());
    println!("  Shape:  {:?}", info.shape);
    println!(
        "  Offset: {} ({} bytes)",
        format_size(info.data_offset),
        info.data_offset
    );
    println!(
        "  Size:   {} ({} bytes)",
        format_size(info.data_size),
        info.data_size
    );
    println!("  Access: zero-copy mmap view");

    if !view.is_empty() {
        let preview = &view[..view.len().min(32)];
        println!("  First {} bytes: {:02x?}", preview.len(), preview);
    }
    Ok(())
}

fn cmd_runtime_slice(args: &RuntimeSliceArgs) -> CliResult {
    let rt = open_runtime(&args.path)?;
    let info = rt
        .tensor_info(&args.name)
        .map_err(|e| format!("tensor '{}' not found: {e}", args.name))?;

    if let Some(bytes_str) = &args.bytes {
        let (off, sz) = parse_pair_u64(bytes_str, "--bytes")?;
        let end = off
            .checked_add(sz)
            .ok_or_else(|| "--bytes offset + size overflows u64".to_string())?;
        let view = rt
            .tensor_byte_view(&args.name, off as usize..end as usize)
            .map_err(|e| format!("failed to read bytes {off}-{end}: {e}"))?;
        println!("Tensor: {} bytes {}..{}", args.name, off, end);
        print_preview(view);
        return Ok(());
    }

    let rows_str = args.rows.as_deref().unwrap_or("0,1");
    let (start, end) = parse_pair_usize(rows_str, "--rows")?;
    let view = rt
        .tensor_rows(&args.name, start, end)
        .map_err(|e| format!("failed to read rows {start}-{end}: {e}"))?;
    let num_rows = end.saturating_sub(start);
    let cols = if info.shape.len() >= 2 {
        info.shape[1]
    } else {
        1
    };
    let elem_size = info.dtype.size_in_bytes();
    println!(
        "Tensor: {} rows {}..{} ({} rows x {} cols x {} bytes/elem)",
        args.name, start, end, num_rows, cols, elem_size
    );
    println!("  Size: {} bytes", view.len());
    print_preview(view);
    Ok(())
}

fn cmd_runtime_stats(args: &RuntimeStatsArgs) -> CliResult {
    let rt = open_runtime(&args.path)?;
    let stats = rt.stats();
    println!("=== Axon Runtime Stats ===");
    println!("File:           {}", args.path.display());
    println!("Model:          {}", rt.model_name());
    println!("Tensor count:   {}", rt.tensor_count());
    println!(
        "Payload size:   {} ({:.2} MB)",
        rt.payload_size(),
        rt.payload_size() as f64 / 1_048_576.0
    );
    println!(
        "File size:      {} ({:.2} MB)",
        rt.file_size(),
        rt.file_size() as f64 / 1_048_576.0
    );
    println!("Mmap:           active");
    println!();
    println!("Access stats:");
    println!("  Bytes read:   {}", stats.bytes_read());
    println!("  Access count: {}", stats.tensor_accesses());
    println!("  Tensor count: {}", rt.tensor_count());
    println!();
    println!("Memory:");
    println!(
        "  Mmap window:  {} (all tensors mapped, OS pages on demand)",
        format_size(rt.file_size())
    );
    println!("  In mem:       OS managed (only accessed pages are resident)");
    println!();
    println!("Cache: use CachedRuntime for LRU caching (`AxonRuntime::with_cache`)");
    println!(
        "  To test: axon runtime inspect {} --cache 1GB",
        args.path.display()
    );
    Ok(())
}

fn cmd_runtime_bench(args: &RuntimeBenchArgs) -> CliResult {
    validate_iterations(args.iterations)?;
    println!(
        "Benchmarking runtime: {} ({} iterations)",
        args.path.display(),
        args.iterations
    );

    let start = Instant::now();
    for _ in 0..args.iterations {
        let _rt = open_runtime(&args.path)?;
    }
    let total_open = start.elapsed();
    println!(
        "Open time:      {:?} avg per open",
        total_open / args.iterations
    );

    let rt = open_runtime(&args.path)?;
    let names = rt.tensor_names();
    if names.is_empty() {
        println!("No tensors to benchmark.");
        return Ok(());
    }

    let start = Instant::now();
    for _ in 0..args.iterations {
        rt.tensor_view(names[0])
            .map_err(|e| format!("failed to read first tensor: {e}"))?;
    }
    let first_time = start.elapsed();
    println!(
        "First tensor:   {:?} avg ({})",
        first_time / args.iterations,
        names[0]
    );

    let start = Instant::now();
    for _ in 0..args.iterations {
        for name in &names {
            rt.tensor_view(name)
                .map_err(|e| format!("failed to read tensor {name}: {e}"))?;
        }
    }
    let scan_time = start.elapsed();
    let total = scan_time / args.iterations;
    let per_tensor = total / names.len() as u32;
    println!(
        "Full scan:      {:?} total ({:?} per tensor, {} tensors)",
        total,
        per_tensor,
        names.len()
    );

    let start = Instant::now();
    for _ in 0..args.iterations {
        rt.tensor_byte_view(names[0], 0..64)
            .map_err(|e| format!("failed to read first byte range: {e}"))?;
    }
    let byte_range = start.elapsed();
    println!(
        "Byte range:     {:?} avg (first 64 bytes)",
        byte_range / args.iterations
    );

    let stats = rt.stats();
    println!();
    println!("  Bytes accessed: {}", stats.bytes_read());
    println!("  Access count:   {}", stats.tensor_accesses());
    Ok(())
}

fn read_axon(path: &PathBuf) -> Result<AxonFile, String> {
    let data = fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    AxonFile::from_bytes(data).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn open_runtime(path: &PathBuf) -> Result<AxonRuntime, String> {
    AxonRuntime::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))
}

fn validate_iterations(iterations: u32) -> CliResult {
    if iterations == 0 {
        Err("iterations must be greater than 0".to_string())
    } else {
        Ok(())
    }
}

fn parse_pair_usize(input: &str, flag: &str) -> Result<(usize, usize), String> {
    let range = input.split_once('=').map_or(input, |(_, value)| value);
    let (start, end) = range
        .split_once(',')
        .ok_or_else(|| format!("expected {flag} 'start,end' format"))?;
    let start = start
        .trim()
        .parse()
        .map_err(|e| format!("invalid {flag} start value: {e}"))?;
    let end = end
        .trim()
        .parse()
        .map_err(|e| format!("invalid {flag} end value: {e}"))?;
    Ok((start, end))
}

fn parse_pair_u64(input: &str, flag: &str) -> Result<(u64, u64), String> {
    let range = input.split_once('=').map_or(input, |(_, value)| value);
    let (offset, size) = range
        .split_once(',')
        .ok_or_else(|| format!("expected {flag} 'offset,size' format"))?;
    let offset = offset
        .trim()
        .parse()
        .map_err(|e| format!("invalid {flag} offset value: {e}"))?;
    let size = size
        .trim()
        .parse()
        .map_err(|e| format!("invalid {flag} size value: {e}"))?;
    Ok((offset, size))
}

fn print_preview(data: &[u8]) {
    let preview = if data.len() > 64 { &data[..64] } else { data };
    println!(
        "  First {} bytes: {:02x?}...",
        preview.len(),
        &preview[..preview.len().min(16)]
    );
}

fn parse_size(s: &str) -> usize {
    let s = s.trim().to_lowercase();
    let (num_str, suffix) = if s.ends_with("gb") {
        (&s[..s.len() - 2], 1_073_741_824usize)
    } else if s.ends_with("mb") {
        (&s[..s.len() - 2], 1_048_576usize)
    } else if s.ends_with("kb") {
        (&s[..s.len() - 2], 1024usize)
    } else {
        (s.as_str(), 1usize)
    };
    let num: f64 = num_str.trim().parse().unwrap_or(0.0);
    (num * suffix as f64) as usize
}

fn npy_header(desc: &TensorDescriptor) -> Vec<u8> {
    let dtype = desc.dtype().unwrap_or(DType::F32);
    let shape = desc.shape_vec();
    let s: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let ds = match dtype {
        DType::F32 => "<f4",
        DType::F16 | DType::BF16 => "<f2",
        DType::I32 => "<i4",
        DType::I64 => "<i8",
        DType::U8 => "u1",
        _ => "<f4",
    };
    let h = format!(
        "{{'descr': '{ds}', 'fortran_order': False, 'shape': ({},) }}",
        s.join(", ")
    );
    let mut hb = h.as_bytes().to_vec();
    let padded = (10 + hb.len()).div_ceil(64) * 64;
    hb.extend(std::iter::repeat_n(b' ', padded - 10 - hb.len()));
    let mut r = Vec::new();
    r.extend_from_slice(b"\x93NUMPY");
    r.push(1);
    r.push(0);
    r.extend_from_slice(&(hb.len() as u16).to_le_bytes());
    r.extend_from_slice(&hb);
    r
}

fn format_size(bytes: u64) -> String {
    let units = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut i = 0;
    while size >= 1024.0 && i < units.len() - 1 {
        size /= 1024.0;
        i += 1;
    }
    format!("{:.2} {}", size, units[i])
}

fn truncate_name(name: &str, max: usize) -> String {
    if name.len() <= max {
        name.to_string()
    } else {
        format!("{}...", &name[..max - 3])
    }
}

fn pseudo_random_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as u8)
        .collect()
}
