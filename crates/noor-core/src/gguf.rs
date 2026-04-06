//! GGUF v3 checkpoint format for Noor.
//!
//! Layout:
//!   Header: magic (u32) + version (u32) + tensor_count (u64) + metadata_kv_count (u64)
//!   Metadata: key-value pairs (model config, training state, tokenizer info)
//!   Tensor info: name, n_dims, dims, type, offset
//!   Tensor data: aligned raw tensor data
//!
//! GGUF is self-describing: the checkpoint IS the deployable model.

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GGUFValue {
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    U64(u64),
}

/// GGUF tensor data type.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGUFType {
    F32 = 0,
    F16 = 1,
    // BF16 added in Phase 1
}

impl GGUFType {
    fn bytes_per_element(&self) -> usize {
        match self {
            GGUFType::F32 => 4,
            GGUFType::F16 => 2,
        }
    }

    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GGUFType::F32),
            1 => Some(GGUFType::F16),
            _ => None,
        }
    }
}

/// Tensor metadata (before data).
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    dtype: GGUFType,
    offset: u64, // offset from start of tensor data section
}

/// Save model weights and metadata to GGUF file.
pub fn save_gguf(
    path: &Path,
    tensors: &HashMap<String, Tensor>,
    metadata: &HashMap<String, GGUFValue>,
) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    let tensor_count = tensors.len() as u64;
    let metadata_count = metadata.len() as u64;

    // Write header
    file.write_all(&GGUF_MAGIC.to_le_bytes())?;
    file.write_all(&GGUF_VERSION.to_le_bytes())?;
    file.write_all(&tensor_count.to_le_bytes())?;
    file.write_all(&metadata_count.to_le_bytes())?;

    // Write metadata KV pairs
    for (key, value) in metadata {
        write_string(&mut file, key)?;
        write_value(&mut file, value)?;
    }

    // Sort tensor names for deterministic output
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    // Compute tensor data offsets
    let mut offset: u64 = 0;
    let mut tensor_infos = Vec::new();
    for name in &names {
        let t = &tensors[*name];
        let info = TensorInfo {
            name: (*name).clone(),
            n_dims: t.shape.len() as u32,
            dims: t.shape.iter().map(|&s| s as u64).collect(),
            dtype: GGUFType::F32, // F32 for Phase 0
            offset,
        };
        let data_size = t.numel() as u64 * GGUFType::F32.bytes_per_element() as u64;
        offset = align_offset(offset + data_size, ALIGNMENT);
        tensor_infos.push(info);
    }

    // Write tensor info entries
    for info in &tensor_infos {
        write_string(&mut file, &info.name)?;
        file.write_all(&info.n_dims.to_le_bytes())?;
        for &d in &info.dims {
            file.write_all(&d.to_le_bytes())?;
        }
        file.write_all(&(info.dtype as u32).to_le_bytes())?;
        file.write_all(&info.offset.to_le_bytes())?;
    }

    // Align to ALIGNMENT before tensor data
    let current_pos = file.stream_position()?;
    let aligned_pos = align_offset(current_pos, ALIGNMENT);
    let padding = aligned_pos - current_pos;
    if padding > 0 {
        file.write_all(&vec![0u8; padding as usize])?;
    }

    let data_start = file.stream_position()?;

    // Write tensor data
    for name in &names {
        let t = &tensors[*name];
        // Write f32 data as raw bytes
        for &val in &t.data {
            file.write_all(&val.to_le_bytes())?;
        }
        // Pad to alignment
        let current = file.stream_position()?;
        let relative = current - data_start;
        let aligned = align_offset(relative, ALIGNMENT);
        let pad = aligned - relative;
        if pad > 0 {
            file.write_all(&vec![0u8; pad as usize])?;
        }
    }

    Ok(())
}

/// Load model weights and metadata from GGUF file.
pub fn load_gguf(
    path: &Path,
) -> io::Result<(HashMap<String, GGUFValue>, HashMap<String, Tensor>)> {
    let mut file = std::fs::File::open(path)?;

    // Read header
    let magic = read_u32(&mut file)?;
    if magic != GGUF_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a GGUF file"));
    }
    let version = read_u32(&mut file)?;
    if version != GGUF_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version: {version}"),
        ));
    }
    let tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;

    // Read metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value = read_value(&mut file)?;
        metadata.insert(key, value);
    }

    // Read tensor info
    let mut tensor_infos = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let n_dims = read_u32(&mut file)?;
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(read_u64(&mut file)?);
        }
        let dtype_val = read_u32(&mut file)?;
        let dtype = GGUFType::from_u32(dtype_val).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Unknown dtype: {dtype_val}"))
        })?;
        let offset = read_u64(&mut file)?;
        tensor_infos.push(TensorInfo { name, n_dims, dims, dtype, offset });
    }

    // Align to find tensor data start
    let current_pos = file.stream_position()?;
    let data_start = align_offset(current_pos, ALIGNMENT);
    file.seek(SeekFrom::Start(data_start))?;

    // Read tensor data
    let mut tensors = HashMap::new();
    for info in &tensor_infos {
        file.seek(SeekFrom::Start(data_start + info.offset))?;
        let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let numel: usize = shape.iter().product();

        let data = match info.dtype {
            GGUFType::F32 => {
                let mut buf = vec![0u8; numel * 4];
                file.read_exact(&mut buf)?;
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>()
            }
            GGUFType::F16 => {
                // Read f16, convert to f32
                let mut buf = vec![0u8; numel * 2];
                file.read_exact(&mut buf)?;
                buf.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect::<Vec<f32>>()
            }
        };

        tensors.insert(info.name.clone(), Tensor::from_slice(&data, &shape));
    }

    Ok((metadata, tensors))
}

/// Collect all named tensors from a NoorModel.
pub fn collect_model_tensors(model: &crate::model::NoorModel) -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();

    // Embedding
    tensors.insert("embedding.weight".to_string(), model.embedding.weight.clone());

    // Output projection
    tensors.insert("output_proj".to_string(), model.output_proj.clone());

    // Final norm
    tensors.insert("final_norm.weight".to_string(), model.final_norm.weight.clone());

    // Blocks
    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blocks.{i}");
        match block {
            crate::layers::block::Block::MoE(b) => {
                // Attention norms
                tensors.insert(format!("{prefix}.attn_norm.pre.weight"), b.attn_norm.pre_norm.weight.clone());
                tensors.insert(format!("{prefix}.attn_norm.post.weight"), b.attn_norm.post_norm.weight.clone());
                tensors.insert(format!("{prefix}.ffn_norm.pre.weight"), b.ffn_norm.pre_norm.weight.clone());
                tensors.insert(format!("{prefix}.ffn_norm.post.weight"), b.ffn_norm.post_norm.weight.clone());
                // Attention weights
                tensors.insert(format!("{prefix}.attn.wq"), b.attention.wq.clone());
                tensors.insert(format!("{prefix}.attn.wk"), b.attention.wk.clone());
                tensors.insert(format!("{prefix}.attn.wv"), b.attention.wv.clone());
                tensors.insert(format!("{prefix}.attn.wo"), b.attention.wo.clone());
                // Dense FFN
                tensors.insert(format!("{prefix}.dense.w_gate"), b.parallel_ffn.dense.w_gate.clone());
                tensors.insert(format!("{prefix}.dense.w_up"), b.parallel_ffn.dense.w_up.clone());
                tensors.insert(format!("{prefix}.dense.w_down"), b.parallel_ffn.dense.w_down.clone());
                // MoE router
                tensors.insert(format!("{prefix}.moe.router.gate"), b.parallel_ffn.moe.router.gate.clone());
                tensors.insert(format!("{prefix}.moe.router.biases"), b.parallel_ffn.moe.router.expert_biases.clone());
                tensors.insert(format!("{prefix}.moe.router.scales"), b.parallel_ffn.moe.router.expert_scales.clone());
                // Experts
                for (j, expert) in b.parallel_ffn.moe.experts.iter().enumerate() {
                    tensors.insert(format!("{prefix}.moe.experts.{j}.w_gate"), expert.w_gate.clone());
                    tensors.insert(format!("{prefix}.moe.experts.{j}.w_up"), expert.w_up.clone());
                    tensors.insert(format!("{prefix}.moe.experts.{j}.w_down"), expert.w_down.clone());
                }
                // Shared expert
                if let Some(ref shared) = b.parallel_ffn.moe.shared_expert {
                    tensors.insert(format!("{prefix}.moe.shared.w_gate"), shared.w_gate.clone());
                    tensors.insert(format!("{prefix}.moe.shared.w_up"), shared.w_up.clone());
                    tensors.insert(format!("{prefix}.moe.shared.w_down"), shared.w_down.clone());
                }
            }
            crate::layers::block::Block::PLE(b) => {
                tensors.insert(format!("{prefix}.attn_norm.weight"), b.attn_norm.weight.clone());
                tensors.insert(format!("{prefix}.ffn_norm.weight"), b.ffn_norm.weight.clone());
                tensors.insert(format!("{prefix}.attn.wq"), b.attention.wq.clone());
                tensors.insert(format!("{prefix}.attn.wk"), b.attention.wk.clone());
                tensors.insert(format!("{prefix}.attn.wv"), b.attention.wv.clone());
                tensors.insert(format!("{prefix}.attn.wo"), b.attention.wo.clone());
                tensors.insert(format!("{prefix}.ffn.w_gate"), b.ffn.w_gate.clone());
                tensors.insert(format!("{prefix}.ffn.w_up"), b.ffn.w_up.clone());
                tensors.insert(format!("{prefix}.ffn.w_down"), b.ffn.w_down.clone());
            }
        }
    }

    // PLE
    if let Some(ref ple) = model.ple {
        tensors.insert("ple.embeddings".to_string(), ple.embeddings.clone());
        tensors.insert("ple.w_gate".to_string(), ple.w_gate.clone());
        tensors.insert("ple.w_up".to_string(), ple.w_up.clone());
    }

    tensors
}

// ---- Wire format helpers ----

fn write_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    let len = s.len() as u64;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_value(w: &mut impl Write, v: &GGUFValue) -> io::Result<()> {
    match v {
        GGUFValue::U32(x) => { w.write_all(&0u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GGUFValue::I32(x) => { w.write_all(&1u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GGUFValue::F32(x) => { w.write_all(&2u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GGUFValue::Bool(x) => { w.write_all(&3u32.to_le_bytes())?; w.write_all(&[*x as u8])?; }
        GGUFValue::String(x) => { w.write_all(&4u32.to_le_bytes())?; write_string(w, x)?; }
        GGUFValue::F64(x) => { w.write_all(&5u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        GGUFValue::U64(x) => { w.write_all(&6u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
    }
    Ok(())
}

fn read_value(r: &mut impl Read) -> io::Result<GGUFValue> {
    let type_id = read_u32(r)?;
    match type_id {
        0 => Ok(GGUFValue::U32(read_u32(r)?)),
        1 => Ok(GGUFValue::I32(read_i32(r)?)),
        2 => Ok(GGUFValue::F32(read_f32(r)?)),
        3 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(GGUFValue::Bool(b[0] != 0)) }
        4 => Ok(GGUFValue::String(read_string(r)?)),
        5 => Ok(GGUFValue::F64(read_f64(r)?)),
        6 => Ok(GGUFValue::U64(read_u64(r)?)),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown value type: {type_id}"))),
    }
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) / alignment * alignment
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normal f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        // Inf or NaN
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }
    // Normal
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("noor_test_{name}.gguf"))
    }

    #[test]
    fn test_save_load_roundtrip() {
        let path = temp_path("roundtrip");
        let mut tensors = HashMap::new();
        tensors.insert("weight_a".to_string(), Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        tensors.insert("weight_b".to_string(), Tensor::from_slice(&[5.0, 6.0, 7.0], &[3]));

        let mut metadata = HashMap::new();
        metadata.insert("model.name".to_string(), GGUFValue::String("noor-test".to_string()));
        metadata.insert("model.layers".to_string(), GGUFValue::U32(16));
        metadata.insert("training.lr".to_string(), GGUFValue::F64(3e-4));

        save_gguf(&path, &tensors, &metadata).expect("Save failed");
        let (loaded_meta, loaded_tensors) = load_gguf(&path).expect("Load failed");

        // Check metadata
        assert!(matches!(loaded_meta.get("model.name"), Some(GGUFValue::String(s)) if s == "noor-test"));
        assert!(matches!(loaded_meta.get("model.layers"), Some(GGUFValue::U32(16))));

        // Check tensors
        let a = &loaded_tensors["weight_a"];
        assert_eq!(a.shape, vec![2, 2]);
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0]);

        let b = &loaded_tensors["weight_b"];
        assert_eq!(b.shape, vec![3]);
        assert_eq!(b.data, vec![5.0, 6.0, 7.0]);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_large_tensor() {
        let path = temp_path("large");
        let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();
        let mut tensors = HashMap::new();
        tensors.insert("big".to_string(), Tensor::from_slice(&data, &[64, 64]));

        save_gguf(&path, &tensors, &HashMap::new()).expect("Save failed");
        let (_, loaded) = load_gguf(&path).expect("Load failed");

        let big = &loaded["big"];
        assert_eq!(big.shape, vec![64, 64]);
        for i in 0..4096 {
            assert!(
                (big.data[i] - data[i]).abs() < 1e-7,
                "Mismatch at {i}: {} vs {}", big.data[i], data[i]
            );
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_model_save_load_roundtrip() {
        use crate::config::ModelConfig;

        // Use a tiny config for speed
        let path = temp_path("model_rt");
        let config_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("config/proxy.toml");
        let mut config = ModelConfig::from_toml(&config_path).unwrap();
        // Override to make it tiny for test speed
        config.model.d_model = 32;
        config.model.n_layers = 2;
        config.model.n_heads = 2;
        config.model.n_kv_heads = 1;
        config.model.head_dim = 16;
        config.model.vocab_size = 100;
        config.moe.n_experts = 4;
        config.moe.n_active_experts = 2;
        config.moe.expert_ffn_dim = 16;
        config.moe.dense_ffn_dim = 32;

        let model = crate::model::NoorModel::from_config(&config);
        let tensors = collect_model_tensors(&model);
        let mut meta = HashMap::new();
        meta.insert("model.name".to_string(), GGUFValue::String("noor-proxy".to_string()));
        meta.insert("training.step".to_string(), GGUFValue::U64(0));

        save_gguf(&path, &tensors, &meta).expect("Save failed");
        let (loaded_meta, loaded_tensors) = load_gguf(&path).expect("Load failed");

        // Verify tensor count matches
        assert_eq!(loaded_tensors.len(), tensors.len(),
            "Tensor count mismatch: saved {}, loaded {}", tensors.len(), loaded_tensors.len());

        // Verify key tensors are bit-exact
        let orig_emb = &tensors["embedding.weight"];
        let load_emb = &loaded_tensors["embedding.weight"];
        assert_eq!(orig_emb.shape, load_emb.shape);
        assert_eq!(orig_emb.data, load_emb.data, "Embedding not bit-exact after roundtrip");

        assert!(matches!(loaded_meta.get("model.name"), Some(GGUFValue::String(s)) if s == "noor-proxy"));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_f16_conversion() {
        // Test known values
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 1e-7); // +0
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-3); // 1.0
        assert!((f16_to_f32(0xC000) - (-2.0)).abs() < 1e-3); // -2.0
        assert!(f16_to_f32(0x7C00).is_infinite()); // +inf
        assert!(f16_to_f32(0x7E00).is_nan()); // NaN
    }
}
