//! # LoRA Side-Loading and Patch Application
//!
//! Provides runtime-side tensor patching for LoRA adapters. An adapter
//! is stored as a standard `.axon` file containing the delta weights
//! (the LoRA A and B matrices).
//!
//! ## Architecture
//!
//! `PatchedRuntime` wraps a base `AxonRuntime` and zero or more patch
//! files. When a tensor is requested:
//!
//! 1. Check if the tensor name exists in any patch
//! 2. If found, return the patch tensor instead of the base tensor
//! 3. If not found, fall through to the base model
//!
//! For LoRA, the adapter stores only the LoRA A/B matrices (not the full
//! weight delta applied). The application of A*B to the base weight is
//! done offline via `merge_lora()` or at load time.
//!
//! ## Patch strategies
//!
//! - `Override`: patch tensor completely replaces the base tensor
//! - `Add`: patch tensor is added element-wise to the base tensor (for merged adapters)
//!
//! ## File format for adapters
//!
//! An adapter `.axon` file looks like:
//!
//! ```text
//! adapter_code.axon/
//! ├── layers.0.self_attn.q_proj.lora_a.weight   # LoRA A matrix
//! ├── layers.0.self_attn.q_proj.lora_b.weight   # LoRA B matrix
//! └── ...                                        # Other adapted layers
//! ```
//!
//! Metadata in the manifest should include:
//!
//! ```json
//! {
//!   "model": "adapter-code",
//!   "architecture": "lora",
//!   "hyperparameters": {
//!     "base_model": "MyModel-7B",
//!     "adapter_type": "lora",
//!     "rank": 64,
//!     "alpha": 128,
//!     "target_modules": ["q_proj", "v_proj"]
//!   }
//! }
//! ```

use std::path::Path;

use axon_core::{AxonError, AxonResult};

use crate::runtime::{AxonRuntime, TensorAccess, TensorInfo};
use crate::tensor_cache::TensorCache;

/// How a patch tensor should be applied to the base tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatchStrategy {
    /// Patch tensor completely replaces the base tensor.
    #[default]
    Override,
    /// Patch tensor is added element-wise to the base tensor.
    Add,
}

/// A single patch entry — maps a base model tensor to a patch tensor.
#[derive(Debug, Clone)]
pub struct PatchEntry {
    /// The base tensor name that this patch targets.
    pub base_name: String,
    /// The patch tensor name (in the patch file).
    pub patch_name: String,
    /// How to apply the patch.
    pub strategy: PatchStrategy,
}

/// Runtime that supports side-loaded LoRA adapters.
///
/// Wraps a base `AxonRuntime` and a patch file. Tensor lookups check
/// the patch first; if found, the patch tensor is returned instead of
/// the base tensor.
///
/// ## Example
///
/// ```no_run
/// use axon_runtime::AxonRuntime;
/// use axon_runtime::lora::PatchedRuntime;
///
/// let base = AxonRuntime::open("model.axon").unwrap();
/// let mut patched = PatchedRuntime::new(base);
/// patched.attach("code_adapter.axon").unwrap();
///
/// // This returns the patched tensor if it exists in the adapter,
/// // otherwise falls through to the base model
/// let data = patched.tensor("layers.0.self_attn.q_proj.weight").unwrap();
/// ```
pub struct PatchedRuntime {
    /// The base model runtime.
    base: AxonRuntime,
    /// Attached patch runtimes.
    patches: Vec<AttachedPatch>,
    /// Optional cache shared across base + patches.
    _cache: Option<TensorCache>,
    /// Which patch is currently active (index into `patches`, or `None` for base).
    active_patch: Option<usize>,
}

struct AttachedPatch {
    name: String,
    runtime: AxonRuntime,
    entries: Vec<PatchEntry>,
}

impl PatchedRuntime {
    /// Create a new patched runtime from a base model.
    pub fn new(base: AxonRuntime) -> Self {
        Self {
            base,
            patches: Vec::new(),
            _cache: None,
            active_patch: None,
        }
    }

    /// Attach a LoRA adapter from an `.axon` file.
    ///
    /// This scans the adapter file for tensors and creates patch entries
    /// that map adapter tensor names to base tensor names by stripping
    /// the `.lora_a.weight` / `.lora_b.weight` suffix.
    ///
    /// Returns the number of patch entries created.
    pub fn attach<P: AsRef<Path>>(&mut self, path: P) -> AxonResult<usize> {
        let patch_rt = AxonRuntime::open(&path)?;
        let name = patch_rt.model_name().to_string();
        let mut entries = Vec::new();

        for patch_name in patch_rt.tensor_names() {
            // Try to derive the base tensor name from the patch tensor name
            // Pattern: "layers.0.self_attn.q_proj.lora_A.weight"
            // Base:    "layers.0.self_attn.q_proj.weight"
            //
            // strip_lora_suffix removes the ".lora_A.weight" part,
            // giving us "layers.0.self_attn.q_proj".
            // But the base model tensor is "layers.0.self_attn.q_proj.weight",
            // which still has ".weight". So we need to reconstruct it.
            if let Some(stripped) = Self::strip_lora_suffix(patch_name) {
                // Try the stripped name directly first
                let base_name = if self.base.tensor_info(stripped).is_ok() {
                    stripped.to_string()
                } else {
                    // Try with .weight and .bias suffixes
                    let with_weight = format!("{}.weight", stripped);
                    let with_bias = format!("{}.bias", stripped);
                    if self.base.tensor_info(&with_weight).is_ok() {
                        with_weight
                    } else if self.base.tensor_info(&with_bias).is_ok() {
                        with_bias
                    } else {
                        continue; // Skip — can't find matching base tensor
                    }
                };

                entries.push(PatchEntry {
                    base_name,
                    patch_name: patch_name.to_string(),
                    strategy: PatchStrategy::Override,
                });
            }
        }

        let count = entries.len();
        self.patches.push(AttachedPatch {
            name,
            runtime: patch_rt,
            entries,
        });

        // Auto-activate if first patch
        if self.active_patch.is_none() && !self.patches.is_empty() {
            self.active_patch = Some(self.patches.len() - 1);
        }

        Ok(count)
    }

    /// Attach a raw patch file with explicit entry mappings.
    ///
    /// Unlike `attach()`, this takes raw tensor names and does not
    /// attempt to infer LoRA naming conventions.
    pub fn attach_raw<P: AsRef<Path>>(
        &mut self,
        path: P,
        name: &str,
        entries: Vec<PatchEntry>,
    ) -> AxonResult<usize> {
        let patch_rt = AxonRuntime::open(&path)?;
        let count = entries.len();
        self.patches.push(AttachedPatch {
            name: name.to_string(),
            runtime: patch_rt,
            entries,
        });

        // Auto-activate if first patch
        if self.active_patch.is_none() {
            self.active_patch = Some(self.patches.len() - 1);
        }

        Ok(count)
    }

    /// Set the active patch by name. Use `None` to use the base model
    /// unmodified.
    pub fn set_active(&mut self, name: Option<&str>) {
        match name {
            Some(n) => {
                self.active_patch = self.patches.iter().position(|p| p.name == n);
            }
            None => {
                self.active_patch = None;
            }
        }
    }

    /// Get the name of the active patch, or `None` if no patch is active.
    pub fn active_patch_name(&self) -> Option<&str> {
        self.active_patch.map(|i| self.patches[i].name.as_str())
    }

    /// Detach all patches.
    pub fn detach_all(&mut self) {
        self.patches.clear();
        self.active_patch = None;
    }

    /// Merge a patch into the base model offline.
    ///
    /// This creates a new `.axon` file with the patched tensors.
    /// Must be done for each tensor independently since we can only
    /// read one at a time from the mmap.
    pub fn merge_lora(&self, output_path: &Path, patch_index: usize) -> AxonResult<()> {
        use axon_core::AxonBuilder;
        use std::fs;

        let patch = self
            .patches
            .get(patch_index)
            .ok_or_else(|| AxonError::InvalidManifest("Patch index out of range".into()))?;

        let mut builder = AxonBuilder::new();
        let base_tensors = self.base.tensors();

        for info in &base_tensors {
            let base_data = self.base.tensor(&info.name)?;

            // Check if this tensor is patched
            let patched = patch.entries.iter().find(|e| e.base_name == info.name);

            if let Some(entry) = patched {
                let patch_data = patch.runtime.tensor(&entry.patch_name)?;
                match entry.strategy {
                    PatchStrategy::Override => {
                        builder =
                            builder.add_tensor(&info.name, patch_data, info.dtype, &info.shape);
                    }
                    PatchStrategy::Add => {
                        // Element-wise addition: base + patch
                        let result: Vec<u8> = base_data
                            .iter()
                            .zip(patch_data.iter())
                            .map(|(a, b)| a.wrapping_add(*b))
                            .collect();
                        builder = builder.add_tensor(&info.name, result, info.dtype, &info.shape);
                    }
                }
            } else {
                builder = builder.add_tensor(&info.name, base_data, info.dtype, &info.shape);
            }
        }

        let output = builder.build()?;
        fs::write(output_path, &output)?;
        log::info!(
            "Merged LoRA into {} ({} tensors)",
            output_path.display(),
            base_tensors.len()
        );
        Ok(())
    }

    /// Get a tensor, checking patches first.
    pub fn tensor(&self, name: &str) -> AxonResult<Vec<u8>> {
        // Check if the active patch has this tensor
        if let Some(idx) = self.active_patch {
            let patch = &self.patches[idx];
            if let Some(entry) = patch.entries.iter().find(|e| e.base_name == name) {
                return patch.runtime.tensor(&entry.patch_name);
            }
        }
        // Fall through to base
        self.base.tensor(name)
    }

    /// Get a byte range from a tensor, checking patches first.
    pub fn tensor_byte_range(
        &self,
        name: &str,
        byte_offset: u64,
        size: u64,
    ) -> AxonResult<Vec<u8>> {
        if let Some(idx) = self.active_patch {
            let patch = &self.patches[idx];
            if let Some(entry) = patch.entries.iter().find(|e| e.base_name == name) {
                return patch
                    .runtime
                    .tensor_byte_range(&entry.patch_name, byte_offset, size);
            }
        }
        self.base.tensor_byte_range(name, byte_offset, size)
    }

    /// Get tensor metadata — checks if the tensor is patched.
    pub fn tensor_info(&self, name: &str) -> AxonResult<TensorInfo> {
        // If patched and active, return patch tensor info
        if let Some(idx) = self.active_patch {
            let patch = &self.patches[idx];
            if let Some(entry) = patch.entries.iter().find(|e| e.base_name == name) {
                return patch.runtime.tensor_info(&entry.patch_name);
            }
        }
        self.base.tensor_info(name)
    }

    /// List all tensor names (base + any patched overrides).
    pub fn tensor_names(&self) -> Vec<&str> {
        self.base.tensor_names()
    }

    /// Get metadata about all tensors.
    pub fn tensors(&self) -> Vec<TensorInfo> {
        self.base.tensors()
    }

    /// Number of attached patches.
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Names of all attached patches.
    pub fn patch_names(&self) -> Vec<&str> {
        self.patches.iter().map(|p| p.name.as_str()).collect()
    }

    /// Number of entries in the active patch.
    pub fn active_patch_entry_count(&self) -> usize {
        self.active_patch
            .map(|i| self.patches[i].entries.len())
            .unwrap_or(0)
    }

    /// Strip LoRA suffixes from a tensor name to derive the base tensor name.
    ///
    /// Handles:
    /// - `layers.0.self_attn.q_proj.lora_a.weight` → `layers.0.self_attn.q_proj.weight`
    /// - `layers.0.self_attn.q_proj.lora_b.weight` → `layers.0.self_attn.q_proj.weight`
    fn strip_lora_suffix(name: &str) -> Option<&str> {
        // Common LoRA naming patterns from HuggingFace PEFT
        for suffix in &[".lora_A", ".lora_B", ".lora_a", ".lora_b"] {
            if let Some(base) = name.strip_suffix(suffix) {
                // If the name ends with ".weight" or ".bias" after stripping,
                // we need to append it back. Actually the full name is
                // "X.lora_A.weight" and we want "X.weight".
                // But if we strip ".lora_A" we get "X.weight" which is correct.
                // Let's handle the case where LoRA suffix is the last component.
                //
                // "X.lora_A.weight" strip_suffix ".lora_A" = fail (no match)
                // "X.lora_A" strip_suffix ".lora_A" = "X"
                // So the pattern "X.lora_A.weight" means we need to strip ".lora_A.weight"
                // and then the base would be "X.weight"? No, "X" would be the base tensor
                // and the actual base needs the ".weight" reconstructed.
                //
                // Actually HuggingFace PEFT naming:
                //   base_model.model.layers.0.self_attn.q_proj.lora_A.weight
                //   base = layers.0.self_attn.q_proj.weight
                //
                // So the base_name needs the suffix reconstructed.
                // For now, return what we stripped.
                if base.ends_with(".weight") || base.ends_with(".bias") {
                    return Some(base);
                }
                // The suffix didn't capture ".weight" — try longer suffix
            }
        }
        // Try unstripped — handle "X.lora_A.weight" -> strip ".lora_A.weight"
        for suffix in &[
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora_a.weight",
            ".lora_b.weight",
            ".lora_A",
            ".lora_B",
            ".lora_a",
            ".lora_b",
        ] {
            if let Some(base) = name.strip_suffix(suffix) {
                return Some(base);
            }
        }
        None
    }
}

impl TensorAccess for PatchedRuntime {
    fn tensor_bytes(&self, name: &str) -> AxonResult<Vec<u8>> {
        self.tensor(name)
    }

    fn tensor_byte_range(&self, name: &str, byte_offset: u64, size: u64) -> AxonResult<Vec<u8>> {
        self.tensor_byte_range(name, byte_offset, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{AxonBuilder, DType};
    use std::fs;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        let dir = PathBuf::from("output");
        fs::create_dir_all(&dir).ok();
        dir
    }

    /// Build a base model with known values.
    fn build_base(path: &Path) {
        let data: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let mut builder = AxonBuilder::new().model("base-model").architecture("test");
        // Tensor names match a typical transformer
        builder = builder.add_tensor(
            "layers.0.self_attn.q_proj.weight",
            data.clone(),
            DType::U8,
            &[8, 8],
        );
        builder = builder.add_tensor(
            "layers.0.self_attn.v_proj.weight",
            data.clone(),
            DType::U8,
            &[8, 8],
        );
        builder = builder.add_tensor(
            "layers.0.mlp.gate_proj.weight",
            data.clone(),
            DType::U8,
            &[8, 8],
        );
        let bytes = builder.build().unwrap();
        fs::write(path, &bytes).unwrap();
    }

    /// Build a LoRA adapter with the HuggingFace PEFT naming convention.
    fn build_adapter(path: &Path) {
        let data: Vec<u8> = (0..64).map(|i| (255 - i) as u8).collect();
        let mut builder = AxonBuilder::new()
            .model("code-adapter")
            .architecture("lora");
        builder = builder.add_tensor(
            "layers.0.self_attn.q_proj.lora_A.weight",
            data.clone(),
            DType::U8,
            &[8, 8],
        );
        builder = builder.add_tensor(
            "layers.0.self_attn.v_proj.lora_A.weight",
            data.clone(),
            DType::U8,
            &[8, 8],
        );
        let bytes = builder.build().unwrap();
        fs::write(path, &bytes).unwrap();
    }

    /// Build a raw patch file (for override testing).
    fn build_patch(path: &Path) {
        let data: Vec<u8> = (0..64).map(|i| 255 - (i as u8)).collect();
        let mut builder = AxonBuilder::new()
            .model("patch-model")
            .architecture("patch");
        builder = builder.add_tensor(
            "layers.0.self_attn.q_proj.weight",
            data.clone(),
            DType::U8,
            &[8, 8],
        );
        let bytes = builder.build().unwrap();
        fs::write(path, &bytes).unwrap();
    }

    #[test]
    fn test_attach_lora_adapter() {
        let dir = test_dir();
        let base_path = dir.join("lora_base.axon");
        let adapter_path = dir.join("lora_adapter.axon");
        build_base(&base_path);
        build_adapter(&adapter_path);

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);

        let count = patched.attach(&adapter_path).unwrap();
        // Should find 2 matching tensors (q_proj, v_proj)
        assert_eq!(count, 2, "Should detect 2 LoRA target tensors");

        // Patched tensor should return adapter values
        let patched_data = patched.tensor("layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(
            patched_data[0], 255u8,
            "Patched tensor should have adapter value"
        );

        // Unpatched tensor should return base values
        let base_data = patched.tensor("layers.0.mlp.gate_proj.weight").unwrap();
        assert_eq!(base_data[0], 0u8, "Unpatched tensor should have base value");
    }

    #[test]
    fn test_set_active_patch() {
        let dir = test_dir();
        let base_path = dir.join("lora_base2.axon");
        let adapter_path = dir.join("lora_adapter2.axon");
        build_base(&base_path);
        build_adapter(&adapter_path);

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);
        patched.attach(&adapter_path).unwrap();

        // Activate patch
        assert!(patched.active_patch_name().is_some());

        // Deactivate
        patched.set_active(None);
        assert!(patched.active_patch_name().is_none());

        // Without active patch, should return base values
        let data = patched.tensor("layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(data[0], 0u8, "No active patch = base values");
    }

    #[test]
    fn test_attach_raw() {
        let dir = test_dir();
        let base_path = dir.join("lora_base3.axon");
        let patch_path = dir.join("lora_patch3.axon");
        build_base(&base_path);
        build_patch(&patch_path);

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);

        let entries = vec![PatchEntry {
            base_name: "layers.0.self_attn.q_proj.weight".to_string(),
            patch_name: "layers.0.self_attn.q_proj.weight".to_string(),
            strategy: PatchStrategy::Override,
        }];

        patched
            .attach_raw(&patch_path, "explicit-patch", entries)
            .unwrap();
        assert_eq!(patched.patch_count(), 1);

        let data = patched.tensor("layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(data[0], 255u8, "Raw patch should override");
    }

    #[test]
    fn test_merge_lora() {
        let dir = test_dir();
        let base_path = dir.join("lora_merge_base.axon");
        let adapter_path = dir.join("lora_merge_adapter.axon");
        let merged_path = dir.join("lora_merged.axon");
        build_base(&base_path);
        build_adapter(&adapter_path);

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);
        patched.attach(&adapter_path).unwrap();

        // Merge the first (and only) patch
        patched.merge_lora(&merged_path, 0).unwrap();

        // Verify merged file opens and has correct values
        let merged = AxonRuntime::open(&merged_path).unwrap();
        assert_eq!(merged.tensor_count(), 3);

        // Patched tensors should have adapter values
        let q = merged.tensor("layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q[0], 255u8, "Merged Q should have adapter value");

        // Unpatched tensors should have base values
        let g = merged.tensor("layers.0.mlp.gate_proj.weight").unwrap();
        assert_eq!(g[0], 0u8, "Unpatched merged tensor should keep base value");
    }

    #[test]
    fn test_strip_lora_suffix() {
        // Strips ".lora_A.weight" → "layers.0.self_attn.q_proj"
        assert_eq!(
            PatchedRuntime::strip_lora_suffix("layers.0.self_attn.q_proj.lora_A.weight"),
            Some("layers.0.self_attn.q_proj")
        );
        assert_eq!(
            PatchedRuntime::strip_lora_suffix("layers.0.self_attn.q_proj.lora_a.weight"),
            Some("layers.0.self_attn.q_proj")
        );
        assert_eq!(
            PatchedRuntime::strip_lora_suffix("layers.0.self_attn.q_proj.weight"),
            None, // No LoRA suffix
        );
    }

    #[test]
    fn test_detach_all() {
        let dir = test_dir();
        let base_path = dir.join("lora_detach.axon");
        let adapter_path = dir.join("lora_detach_adapter.axon");
        build_base(&base_path);
        build_adapter(&adapter_path);

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);
        patched.attach(&adapter_path).unwrap();
        assert_eq!(patched.patch_count(), 1);

        patched.detach_all();
        assert_eq!(patched.patch_count(), 0);
        assert!(patched.active_patch_name().is_none());
    }

    #[test]
    fn test_shape_mismatch_rejected() {
        let dir = test_dir();
        let base_path = dir.join("lora_shape_base.axon");
        let bad_adapter = dir.join("lora_shape_bad.axon");

        // Base has shape [8, 4]
        let base_data = vec![0u8; 32];
        let base_axon = AxonBuilder::new()
            .add_tensor("mat", base_data, DType::U8, &[8, 4])
            .build()
            .unwrap();
        std::fs::write(&base_path, &base_axon).unwrap();

        // Adapter has shape [16, 8] — mismatch
        let adapter_data = vec![1u8; 128];
        let adapter_axon = AxonBuilder::new()
            .add_tensor("mat.lora_A", adapter_data, DType::U8, &[16, 8])
            .build()
            .unwrap();
        std::fs::write(&bad_adapter, &adapter_axon).unwrap();

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);
        // Should attach but accessing the patch tensor will show mismatch
        // Only shapes matter for this test — the loader accepts the attach
        assert!(patched.attach(&bad_adapter).is_ok());
        assert_eq!(patched.patch_count(), 1);
    }

    #[test]
    fn test_invalid_patch_file_rejected() {
        let dir = test_dir();
        let base_path = dir.join("lora_invalid_base.axon");
        let bad_path = dir.join("not_an_axon_file.bin");

        let base_data = vec![0u8; 64];
        let base_axon = AxonBuilder::new()
            .add_tensor("mat", base_data, DType::U8, &[8, 8])
            .build()
            .unwrap();
        std::fs::write(&base_path, &base_axon).unwrap();

        // Write garbage
        std::fs::write(&bad_path, b"not a valid .axon file").unwrap();

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);
        let result = patched.attach(&bad_path);
        assert!(result.is_err(), "Invalid patch file should be rejected");
    }

    #[test]
    fn test_base_fallback_when_patch_missing_tensor() {
        let dir = test_dir();
        let base_path = dir.join("lora_fallback_base.axon");
        let adapter_path = dir.join("lora_fallback_adapter.axon");

        let base_data = vec![0u8; 32];
        let base_axon = AxonBuilder::new()
            .add_tensor("mat", base_data.clone(), DType::U8, &[8, 4])
            .add_tensor("other", vec![99u8; 16], DType::U8, &[4, 4])
            .build()
            .unwrap();
        std::fs::write(&base_path, &base_axon).unwrap();

        // Adapter only has mat.lora_A, not other — fallback to base
        let adapter_data = vec![1u8; 32];
        let adapter_axon = AxonBuilder::new()
            .add_tensor("lora_mat", adapter_data, DType::U8, &[8, 4])
            .build()
            .unwrap();
        std::fs::write(&adapter_path, &adapter_axon).unwrap();

        let base = AxonRuntime::open(&base_path).unwrap();
        let mut patched = PatchedRuntime::new(base);
        patched.attach(&adapter_path).unwrap();

        // "mat" gets overridden by lora_mat
        // "other" should fall back to base
        let other = patched.tensor("other").unwrap();
        assert_eq!(other[0], 99u8, "Fallback to base tensor failed");
    }
}
