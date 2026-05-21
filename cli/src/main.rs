use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use axon_core::*;
use axon_infer::chat::InferenceEngine;
use axon_infer::sampling::SamplingParams;
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
    /// Import a GGUF model into Axon format
    ImportGguf(ImportGgufArgs),
    /// Import a locally installed Ollama model into Axon format
    ImportOllama(ImportOllamaArgs),
    /// Chat through a local Ollama model and optionally print performance stats
    Chat(ChatArgs),
    /// Benchmark Ollama chat speed, load time, and memory for a model
    BenchOllama(BenchOllamaArgs),
    /// Run a model natively from an .axon file (no Ollama needed)
    Run(RunArgs),
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
struct ImportGgufArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Args)]
struct ImportOllamaArgs {
    model: String,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    models_dir: Option<PathBuf>,
}

#[derive(Args)]
struct ChatArgs {
    model: String,
    prompt: Vec<String>,
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    endpoint: String,
    #[arg(long, default_value = "5m")]
    keep_alive: String,
    #[arg(long)]
    stats: bool,
}

#[derive(Args)]
struct RunArgs {
    /// Path to the .axon model file
    model: PathBuf,
    /// Prompt to send (if not provided, starts interactive chat)
    prompt: Option<String>,
    /// System prompt for interactive mode
    #[arg(long)]
    system_prompt: Option<String>,
    /// Maximum number of tokens to generate
    #[arg(long, default_value = "512")]
    max_tokens: usize,
    /// Temperature (0 = greedy, higher = more random)
    #[arg(long, default_value = "0.7")]
    temperature: f32,
    /// Top-k sampling
    #[arg(long, default_value = "40")]
    top_k: usize,
    /// Top-p (nucleus) sampling
    #[arg(long, default_value = "0.9")]
    top_p: f32,
    /// Repetition penalty
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,
    /// Show detailed generation stats
    #[arg(long)]
    stats: bool,
}

#[derive(Args)]
struct BenchOllamaArgs {
    model: String,
    #[arg(short, long, default_value = "3")]
    runs: u32,
    #[arg(short, long, default_value = "Reply with one short sentence.")]
    prompt: String,
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    endpoint: String,
    #[arg(long, default_value = "5m")]
    keep_alive: String,
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
        Commands::ImportGguf(a) => cmd_import_gguf(&a),
        Commands::ImportOllama(a) => cmd_import_ollama(&a),
        Commands::Chat(a) => cmd_chat(&a),
        Commands::BenchOllama(a) => cmd_bench_ollama(&a),
        Commands::Run(a) => cmd_run(&a),
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

    // Set model and architecture metadata
    let model_name = args.model.as_deref().unwrap_or("test");
    let architecture = args.architecture.as_deref().unwrap_or("llama");
    builder = builder.model(model_name).architecture(architecture);

    // Set hyperparameters needed by the inference engine
    let n_layers = 2u64;
    let dim = 4096u64;
    let n_heads = 32u64;
    let hidden_dim = 11008u64;
    let vocab_size = 32000u64;

    builder = builder
        .metadata("block_count", serde_json::json!(n_layers))
        .metadata("embedding_length", serde_json::json!(dim))
        .metadata("attention.head_count", serde_json::json!(n_heads))
        .metadata("attention.head_count_kv", serde_json::json!(n_heads))
        .metadata("feed_forward_length", serde_json::json!(hidden_dim))
        .metadata("vocab_size", serde_json::json!(vocab_size))
        .metadata("context_length", serde_json::json!(4096))
        .metadata("attention.layer_norm_rms_epsilon", serde_json::json!(1e-5));

    // Use GGUF-compatible tensor names
    for layer in 0..n_layers as u64 {
        for proj in &["attn_q", "attn_k", "attn_v", "attn_output"] {
            builder = builder.add_tensor(
                &format!("blk.{}.{}.weight", layer, proj),
                pseudo_random_bytes(4096 * 4096),
                DType::Q4,
                &[4096, 4096],
            );
        }
    }
    for layer in 0..n_layers as u64 {
        for p in &["ffn_gate", "ffn_up", "ffn_down"] {
            let (rows, cols) = if *p == "ffn_down" {
                (hidden_dim as usize, dim as usize)
            } else {
                (dim as usize, hidden_dim as usize)
            };
            let shape = if *p == "ffn_down" {
                vec![hidden_dim, dim]
            } else {
                vec![dim, hidden_dim]
            };
            builder = builder.add_tensor(
                &format!("blk.{}.{}.weight", layer, p),
                pseudo_random_bytes(rows * cols * 2),
                DType::F16,
                &shape,
            );
        }
        // Norms
        builder = builder.add_tensor(
            &format!("blk.{}.attn_norm.weight", layer),
            pseudo_random_bytes(dim as usize * 2),
            DType::F16,
            &[dim],
        );
        builder = builder.add_tensor(
            &format!("blk.{}.ffn_norm.weight", layer),
            pseudo_random_bytes(dim as usize * 2),
            DType::F16,
            &[dim],
        );
    }

    // Token embeddings
    builder = builder.add_tensor(
        "token_embd.weight",
        pseudo_random_bytes(vocab_size as usize * dim as usize * 2),
        DType::F16,
        &[vocab_size, dim],
    );
    // Output norm
    builder = builder.add_tensor(
        "output_norm.weight",
        pseudo_random_bytes(dim as usize * 2),
        DType::F16,
        &[dim],
    );
    // LM head (output projection)
    builder = builder.add_tensor(
        "output.weight",
        pseudo_random_bytes(vocab_size as usize * dim as usize * 2),
        DType::F16,
        &[vocab_size, dim],
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
        2 + n_layers as u32 * 9
    );
    Ok(())
}

fn cmd_import_gguf(args: &ImportGgufArgs) -> CliResult {
    let axon_bytes = axon_core::convert::gguf_to_axon(&args.input)
        .map_err(|e| format!("failed to import GGUF {}: {e}", args.input.display()))?;
    fs::write(&args.output, &axon_bytes)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
    println!(
        "Imported GGUF {} -> {} ({:.2} MB)",
        args.input.display(),
        args.output.display(),
        axon_bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}

fn cmd_import_ollama(args: &ImportOllamaArgs) -> CliResult {
    let source = axon_core::convert::resolve_ollama_model(&args.model, args.models_dir.as_deref())
        .map_err(|e| format!("failed to resolve Ollama model {}: {e}", args.model))?;
    let axon_bytes =
        axon_core::convert::ollama_model_to_axon(&args.model, args.models_dir.as_deref())
            .map_err(|e| format!("failed to import Ollama model {}: {e}", args.model))?;
    fs::write(&args.output, &axon_bytes)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
    println!(
        "Imported Ollama model {} ({}) -> {} ({:.2} MB)",
        args.model,
        source.blob_path.display(),
        args.output.display(),
        axon_bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}

fn cmd_run(args: &RunArgs) -> CliResult {
    let path = &args.model;

    println!("Loading model: {}", path.display());
    let start = Instant::now();
    let mut engine =
        InferenceEngine::load(path).map_err(|e| format!("failed to load model: {e}"))?;
    let load_time = start.elapsed();

    let config = engine.config();

    println!(
        "  Architecture: {} ({} layers, {} dim, {} heads, {} vocab)",
        config.architecture, config.n_layers, config.dim, config.n_heads, config.vocab_size
    );
    println!(
        "  Context: {} tokens  |  Load time: {:?}",
        config.ctx_len, load_time
    );
    println!();

    // Set sampling params
    engine.set_sampling(SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
    });

    if let Some(prompt) = &args.prompt {
        // Single prompt mode
        print!("{}", prompt);
        std::io::stdout().flush().map_err(|e| format!("IO error: {e}"))?;

        let gen_start = Instant::now();
        let (response, stats) = engine
            .generate_text(prompt, args.max_tokens)
            .map_err(|e| format!("generation failed: {e}"))?;
        let gen_time = gen_start.elapsed().as_secs_f64();
        println!("{}", response);
        if args.stats {
            eprintln!(
                "  [prompt: {} tok | gen: {} tok | {:.2} tok/s | total: {:.1}s]",
                stats.prompt_tokens,
                stats.generated_tokens,
                stats.tokens_per_second,
                gen_time
            );
        }
    } else {
        // Interactive chat mode
        engine
            .chat(args.system_prompt.as_deref(), args.max_tokens)
            .map_err(|e| format!("chat error: {e}"))?;
    }

    Ok(())
}

fn cmd_chat(args: &ChatArgs) -> CliResult {
    let prompt = join_prompt(&args.prompt)?;
    let result = ollama_chat(&args.endpoint, &args.model, &prompt, &args.keep_alive)?;
    println!("{}", result.response.trim());
    if args.stats {
        println!();
        print_ollama_stats("Stats", &result);
        print_ollama_memory(&args.endpoint, &args.model)?;
    }
    Ok(())
}

fn cmd_bench_ollama(args: &BenchOllamaArgs) -> CliResult {
    validate_iterations(args.runs)?;
    println!(
        "Benchmarking Ollama model {} through {} ({} runs)",
        args.model, args.endpoint, args.runs
    );
    println!("Prompt: {}", args.prompt);
    println!();

    let mut total_wall_ms = 0.0;
    let mut total_eval_tokens = 0u64;
    let mut total_eval_ns = 0u64;
    let mut total_prompt_tokens = 0u64;
    let mut total_prompt_ns = 0u64;
    let mut total_load_ns = 0u64;

    for run in 1..=args.runs {
        let wall = Instant::now();
        let result = ollama_chat(&args.endpoint, &args.model, &args.prompt, &args.keep_alive)?;
        let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
        total_wall_ms += wall_ms;
        total_eval_tokens += result.eval_count;
        total_eval_ns += result.eval_duration;
        total_prompt_tokens += result.prompt_eval_count;
        total_prompt_ns += result.prompt_eval_duration;
        total_load_ns += result.load_duration;
        println!(
            "Run {run}: {:.2} tok/s, load {}, wall {:.2} ms",
            tokens_per_second(result.eval_count, result.eval_duration),
            format_ns(result.load_duration),
            wall_ms
        );
    }

    println!();
    println!("Summary:");
    println!(
        "  Output speed:  {:.2} tokens/s",
        tokens_per_second(total_eval_tokens, total_eval_ns)
    );
    println!(
        "  Prompt speed:  {:.2} tokens/s",
        tokens_per_second(total_prompt_tokens, total_prompt_ns)
    );
    println!(
        "  Avg load time: {}",
        format_ns(total_load_ns / args.runs as u64)
    );
    println!(
        "  Avg wall time: {:.2} ms",
        total_wall_ms / args.runs as f64
    );
    print_ollama_memory(&args.endpoint, &args.model)?;
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

#[derive(Debug)]
struct OllamaChatResult {
    response: String,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: u64,
    prompt_eval_duration: u64,
    eval_count: u64,
    eval_duration: u64,
}

fn join_prompt(parts: &[String]) -> Result<String, String> {
    let prompt = parts.join(" ");
    if prompt.trim().is_empty() {
        Err("please provide a prompt, for example: axon chat gemma3:1b \"Hello\"".to_string())
    } else {
        Ok(prompt)
    }
}

fn ollama_chat(
    endpoint: &str,
    model: &str,
    prompt: &str,
    keep_alive: &str,
) -> Result<OllamaChatResult, String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
        "keep_alive": keep_alive,
    });
    let response = http_json(endpoint, "POST", "/api/chat", Some(&body))?;
    let content = response
        .get("message")
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(OllamaChatResult {
        response: content,
        total_duration: response
            .get("total_duration")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        load_duration: response
            .get("load_duration")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        prompt_eval_count: response
            .get("prompt_eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        prompt_eval_duration: response
            .get("prompt_eval_duration")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        eval_count: response
            .get("eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        eval_duration: response
            .get("eval_duration")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    })
}

fn print_ollama_stats(title: &str, result: &OllamaChatResult) {
    println!("{title}:");
    println!("  Total time:    {}", format_ns(result.total_duration));
    println!("  Load time:     {}", format_ns(result.load_duration));
    println!(
        "  Prompt eval:   {} tokens in {} ({:.2} tok/s)",
        result.prompt_eval_count,
        format_ns(result.prompt_eval_duration),
        tokens_per_second(result.prompt_eval_count, result.prompt_eval_duration)
    );
    println!(
        "  Generation:    {} tokens in {} ({:.2} tok/s)",
        result.eval_count,
        format_ns(result.eval_duration),
        tokens_per_second(result.eval_count, result.eval_duration)
    );
}

fn print_ollama_memory(endpoint: &str, model: &str) -> CliResult {
    let ps = http_json(endpoint, "GET", "/api/ps", None)?;
    let Some(models) = ps.get("models").and_then(|v| v.as_array()) else {
        println!("  Memory:        unavailable (/api/ps returned no models)");
        return Ok(());
    };
    let loaded = models.iter().find(|entry| {
        entry
            .get("name")
            .or_else(|| entry.get("model"))
            .and_then(|v| v.as_str())
            .is_some_and(|name| name == model)
    });
    if let Some(entry) = loaded {
        let size = entry.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let vram = entry.get("size_vram").and_then(|v| v.as_u64()).unwrap_or(0);
        let context = entry
            .get("context_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        println!("  Memory size:   {}", format_size(size));
        println!("  VRAM size:     {}", format_size(vram));
        if context > 0 {
            println!("  Context:       {context} tokens");
        }
    } else {
        println!("  Memory:        model is not currently listed by Ollama");
    }
    Ok(())
}

fn http_json(
    endpoint: &str,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let (host, port) = parse_http_endpoint(endpoint)?;
    let body_text = body.map(|v| v.to_string()).unwrap_or_default();
    let request = if body.is_some() {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_text}",
            body_text.len()
        )
    } else {
        format!("{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n")
    };

    let mut stream = TcpStream::connect((host.as_str(), port))
        .map_err(|e| format!("failed to connect to Ollama at {endpoint}: {e}"))?;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("failed to write request to Ollama: {e}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| format!("failed to read Ollama response: {e}"))?;

    let response_text = String::from_utf8_lossy(&response);
    let (headers, raw_body) = response_text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "invalid HTTP response from Ollama".to_string())?;
    if !headers.starts_with("HTTP/1.1 200") && !headers.starts_with("HTTP/1.0 200") {
        return Err(format!("Ollama request failed: {}", raw_body.trim()));
    }
    let body_text = if headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        decode_chunked(raw_body)?
    } else {
        raw_body.to_string()
    };
    serde_json::from_str(&body_text)
        .map_err(|e| format!("failed to parse Ollama JSON response: {e}"))
}

fn parse_http_endpoint(endpoint: &str) -> Result<(String, u16), String> {
    let rest = endpoint
        .strip_prefix("http://")
        .ok_or_else(|| "only http:// Ollama endpoints are supported".to_string())?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port) = authority.rsplit_once(':').map_or((authority, "80"), |v| v);
    let port = port
        .parse()
        .map_err(|e| format!("invalid Ollama endpoint port: {e}"))?;
    Ok((host.to_string(), port))
}

fn decode_chunked(raw: &str) -> Result<String, String> {
    let mut decoded = String::new();
    let mut rest = raw;
    loop {
        let (size_hex, after_size) = rest
            .split_once("\r\n")
            .ok_or_else(|| "invalid chunked Ollama response".to_string())?;
        let size = usize::from_str_radix(size_hex.trim(), 16)
            .map_err(|e| format!("invalid HTTP chunk size: {e}"))?;
        if size == 0 {
            break;
        }
        if after_size.len() < size + 2 {
            return Err("truncated chunked Ollama response".to_string());
        }
        decoded.push_str(&after_size[..size]);
        rest = &after_size[size + 2..];
    }
    Ok(decoded)
}

fn tokens_per_second(tokens: u64, duration_ns: u64) -> f64 {
    if duration_ns == 0 {
        0.0
    } else {
        tokens as f64 / (duration_ns as f64 / 1_000_000_000.0)
    }
}

fn format_ns(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.2} us", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
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
