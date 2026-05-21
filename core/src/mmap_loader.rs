use std::fs::File;
use std::path::Path;

use crate::checksum;
use crate::error::AxonResult;
use crate::header::AxonHeader;
use crate::manifest::{AxonFile, Manifest};
use crate::tensor::{Affinity, DType, TensorDescriptor};

pub struct MappedAxonFile<'a> {
    _mmap: memmap2::Mmap,
    pub file: AxonFile,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> MappedAxonFile<'a> {
    pub fn open<P: AsRef<Path>>(path: P) -> AxonResult<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let axon_file = AxonFile::from_bytes(mmap[..].to_vec())?;
        Ok(Self {
            _mmap: mmap,
            file: axon_file,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn view(&self, offset: u64, size: u64) -> Option<&[u8]> {
        let start = offset as usize;
        let end = start + size as usize;
        if end <= self.file.data.len() {
            Some(&self.file.data[start..end])
        } else {
            None
        }
    }

    pub fn tensor_data_by_name(&self, name: &str) -> Option<&[u8]> {
        self.file.tensor_data(name)
    }

    pub fn summary(&self) -> AxonFileSummary {
        let mut tensor_info = Vec::new();
        for (name, desc) in &self.file.manifest.tensors {
            let dtype = desc.dtype().unwrap_or(DType::F32);
            tensor_info.push(TensorSummary {
                name: name.clone(),
                dtype: dtype.name().to_string(),
                shape: desc.shape_vec(),
                size_bytes: desc.data_size,
            });
        }
        AxonFileSummary {
            model: self.file.manifest.model.clone().unwrap_or_default(),
            architecture: self.file.manifest.architecture.clone().unwrap_or_default(),
            tensor_count: self.file.header.tensor_count,
            payload_size: self.file.header.payload_size,
            tensors: tensor_info,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorSummary {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct AxonFileSummary {
    pub model: String,
    pub architecture: String,
    pub tensor_count: u64,
    pub payload_size: u64,
    pub tensors: Vec<TensorSummary>,
}

pub struct AxonBuilder {
    header: AxonHeader,
    manifest: Manifest,
    tensor_data: Vec<(String, Vec<u8>, DType, Vec<u64>)>,
}

impl AxonBuilder {
    pub fn new() -> Self {
        Self {
            header: AxonHeader::default(),
            manifest: Manifest::new(),
            tensor_data: Vec::new(),
        }
    }
    pub fn model(mut self, name: &str) -> Self {
        self.manifest.model = Some(name.to_string());
        self
    }
    pub fn architecture(mut self, arch: &str) -> Self {
        self.manifest.architecture = Some(arch.to_string());
        self
    }

    pub fn add_tensor(mut self, name: &str, data: Vec<u8>, dtype: DType, shape: &[u64]) -> Self {
        let expected = shape.iter().product::<u64>() as usize * dtype.size_in_bytes();
        assert_eq!(data.len(), expected, "Tensor data size mismatch for {name}");
        self.tensor_data
            .push((name.to_string(), data, dtype, shape.to_vec()));
        self
    }

    pub fn build(mut self) -> AxonResult<Vec<u8>> {
        self.tensor_data.sort_by(|a, b| a.0.cmp(&b.0));

        let mut result = Vec::new();
        result.extend_from_slice(&self.header.to_bytes()?);
        while result.len() < 4096 {
            result.push(0u8);
        }

        let manifest_json = self.manifest.to_json_bytes()?;
        let _manifest_offset = result.len() as u64;
        result.extend_from_slice(&manifest_json);

        let tdt_start = AxonHeader::align_up(result.len() as u64, 64) as usize;
        while result.len() < tdt_start {
            result.push(0u8);
        }

        let tensor_count = self.tensor_data.len() as u64;
        let payload_offset = tdt_start as u64 + tensor_count * TensorDescriptor::SIZE as u64;
        let payload_offset = AxonHeader::align_up(payload_offset, 64);

        let mut descriptors = Vec::new();
        let mut current_offset = 0u64;

        for (name, data, dtype, shape) in &self.tensor_data {
            let data_size = data.len() as u64;
            let aligned_offset = AxonHeader::align_up(payload_offset + current_offset, 64);
            let padding = aligned_offset - (payload_offset + current_offset);
            current_offset += padding;
            let cksum = checksum::xxh3_64(data);
            descriptors.push(TensorDescriptor::new(
                name,
                *dtype,
                shape,
                aligned_offset,
                data_size,
                Affinity::Default,
                cksum,
            ));
            current_offset += data_size;
        }

        for desc in &descriptors {
            result.extend_from_slice(&desc.to_bytes()?);
        }
        while (result.len() as u64) < payload_offset {
            result.push(0u8);
        }

        for (_name, data, _dtype, _shape) in &self.tensor_data {
            let aligned = AxonHeader::align_up(result.len() as u64, 64) as usize;
            while result.len() < aligned {
                result.push(0u8);
            }
            result.extend_from_slice(data);
        }

        let payload_size = result.len() as u64 - payload_offset;
        let finalized_header =
            AxonHeader::new(tensor_count, manifest_json.len() as u64, payload_size);
        result[..AxonHeader::HEADER_SIZE].copy_from_slice(&finalized_header.to_bytes()?);
        Ok(result)
    }
}

impl Default for AxonBuilder {
    fn default() -> Self {
        Self::new()
    }
}
