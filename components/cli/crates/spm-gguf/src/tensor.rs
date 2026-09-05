use crate::TensorInfo;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

const Q6_K_ELEMENTS: usize = 256;
const Q6_K_BYTES: usize = 210;
const Q5_K_BYTES: usize = 176;
const Q8_0_ELEMENTS: usize = 32;
const Q8_0_BYTES: usize = 34;

/// Read one already-validated tensor range, subject to a caller-selected cap.
///
/// The descriptor must come from [`crate::read`]. Keeping the allocation cap
/// explicit prevents metadata from turning this helper into an unbounded read.
/// # Errors
/// Returns an error when the range exceeds the cap, does not fit in memory, or
/// cannot be read exactly.
pub fn read_tensor_bytes(
    path: &Path,
    tensor: &TensorInfo,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    read_tensor_range(path, tensor, 0, tensor.len, max_bytes)
}

/// Read a bounded byte range within one already-validated tensor.
/// # Errors
/// Returns an error when arithmetic or bounds checks fail, the range exceeds
/// the cap, or the file cannot be read exactly.
pub fn read_tensor_range(
    path: &Path,
    tensor: &TensorInfo,
    relative_offset: u64,
    len: u64,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    if len > max_bytes {
        return Err(format!(
            "tensor {} request is {} bytes, exceeding limit {max_bytes}",
            tensor.name, len
        ));
    }
    let relative_end = relative_offset
        .checked_add(len)
        .ok_or("tensor range overflow")?;
    if relative_end > tensor.len {
        return Err(format!("requested range exceeds tensor {}", tensor.name));
    }
    let absolute_offset = tensor
        .offset
        .checked_add(relative_offset)
        .ok_or("tensor offset overflow")?;
    let len = usize::try_from(len).map_err(|_| "tensor is too large for this platform")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| "tensor allocation refused")?;
    bytes.resize(len, 0);
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|e| e.to_string())?;
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// Decode little-endian GGML F32 tensor bytes.
/// # Errors
/// Returns an error unless the input contains whole 32-bit values.
pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err("F32 byte length is not divisible by 4".into());
    }
    Ok(chunks
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let out = match exp {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            let exponent = 113 - shift;
            sign | (exponent << 23) | ((normalized & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exp + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(out)
}

/// Decode GGML `Q6_K` blocks using the reference 256-value, 210-byte layout.
/// # Errors
/// Returns an error unless the input contains whole `Q6_K` blocks.
pub fn decode_q6_k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let blocks = bytes.chunks_exact(Q6_K_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("Q6_K byte length is not divisible by 210".into());
    }
    let mut out = Vec::new();
    out.try_reserve_exact((bytes.len() / Q6_K_BYTES) * Q6_K_ELEMENTS)
        .map_err(|_| "decoded tensor allocation refused")?;
    for block in blocks {
        let (ql, rest) = block.split_at(128);
        let (qh, rest) = rest.split_at(64);
        let (scales, d_bytes) = rest.split_at(16);
        let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));
        for half in 0..2 {
            let ql = &ql[half * 64..];
            let qh = &qh[half * 32..];
            let scales = &scales[half * 8..];
            let base = out.len();
            out.resize(base + 128, 0.0);
            for l in 0..32 {
                let is = l / 16;
                let quants = [
                    (ql[l] & 0x0f) | (((qh[l]) & 3) << 4),
                    (ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4),
                    (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                    (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
                ];
                for lane in 0..4 {
                    let q = f32::from(quants[lane]) - 32.0;
                    let scale = f32::from(i8::from_ne_bytes([scales[is + lane * 2]]));
                    out[base + l + lane * 32] = d * scale * q;
                }
            }
        }
    }
    Ok(out)
}

/// Decode GGML `Q5_K` super-blocks: 256 weights in 176 bytes.
/// # Errors
/// Returns an error unless the input contains whole `Q5_K` blocks.
pub fn decode_q5_k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let blocks = bytes.chunks_exact(Q5_K_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("Q5_K byte length is not divisible by 176".into());
    }
    let mut out = Vec::new();
    out.try_reserve_exact((bytes.len() / Q5_K_BYTES) * Q6_K_ELEMENTS)
        .map_err(|_| "decoded tensor allocation refused")?;
    for block in blocks {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        for group in 0..4 {
            for half in 0..2 {
                let index = group * 2 + half;
                let (scale, minimum) = scale_min_q5(index, scales);
                let bit = 1_u8 << index;
                for lane in 0..32 {
                    let packed = qs[group * 32 + lane];
                    let low = if half == 0 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    let quant = low + if qh[lane] & bit != 0 { 16 } else { 0 };
                    out.push(d * f32::from(scale) * f32::from(quant) - dmin * f32::from(minimum));
                }
            }
        }
    }
    Ok(out)
}

fn scale_min_q5(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

/// Decode GGML `Q8_0` blocks: 32 weights in 34 bytes.
/// # Errors
/// Returns an error unless the input contains whole `Q8_0` blocks.
pub fn decode_q8_0(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let blocks = bytes.chunks_exact(Q8_0_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("Q8_0 byte length is not divisible by 34".into());
    }
    let mut out = Vec::new();
    out.try_reserve_exact((bytes.len() / Q8_0_BYTES) * Q8_0_ELEMENTS)
        .map_err(|_| "decoded tensor allocation refused")?;
    for block in blocks {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        out.extend(
            block[2..]
                .iter()
                .map(|&value| d * f32::from(i8::from_ne_bytes([value]))),
        );
    }
    Ok(out)
}
