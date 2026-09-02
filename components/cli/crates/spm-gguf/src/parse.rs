use crate::{
    Content, MAX_DIMS, MAX_METADATA, MAX_TENSORS, TensorInfo, model::RawTensor, wire::Parser,
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

fn layout(t: u32) -> Option<(u64, u64)> {
    match t {
        0 => Some((1, 4)),
        1 | 30 => Some((1, 2)),
        2 => Some((32, 18)),
        3 => Some((32, 20)),
        6 => Some((32, 22)),
        7 => Some((32, 24)),
        8 => Some((32, 34)),
        9 => Some((32, 40)),
        10 => Some((256, 84)),
        11 => Some((256, 110)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        15 => Some((256, 292)),
        16 => Some((1, 8)),
        _ => None,
    }
}

fn prelude(p: &mut Parser) -> Result<(u32, u64, u64), String> {
    if p.bytes(4)?.as_slice() != b"GGUF" {
        return Err("invalid GGUF magic".into());
    }
    let version = p.u32()?;
    if version != 3 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let tensors = p.u64()?;
    let metadata = p.u64()?;
    if tensors > MAX_TENSORS {
        return Err("tensor count exceeds limit".into());
    }
    if metadata > MAX_METADATA {
        return Err("metadata count exceeds limit".into());
    }
    Ok((version, tensors, metadata))
}

fn metadata(p: &mut Parser, n: u64) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    out.try_reserve(usize::try_from(n).map_err(|_| "metadata count too large")?)
        .map_err(|_| "metadata allocation refused")?;
    for _ in 0..n {
        let key = p.string()?;
        let ty = p.u32()?;
        let value = p.value(ty, 0)?.unwrap_or_else(|| format!("<{ty}>"));
        if out.insert(key, value).is_some() {
            return Err("duplicate metadata key".into());
        }
    }
    Ok(out)
}

fn descriptors(p: &mut Parser, n: u64) -> Result<Vec<RawTensor>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(n).map_err(|_| "tensor count too large")?)
        .map_err(|_| "tensor allocation refused")?;
    let mut names = HashSet::new();
    for _ in 0..n {
        let name = p.string()?;
        if !names.insert(name.clone()) {
            return Err(format!("duplicate tensor name {name}"));
        }
        let nd = p.u32()?;
        if nd == 0 || nd > MAX_DIMS {
            return Err(format!("invalid dimension count {nd}"));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(p.u64()?);
        }
        out.push(RawTensor {
            name,
            dims,
            dtype: p.u32()?,
            offset: p.u64()?,
        });
    }
    Ok(out)
}

fn ranges(
    raw: Vec<RawTensor>,
    base: u64,
    file_len: u64,
    align: u64,
) -> Result<Vec<TensorInfo>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        if r.offset % align != 0 {
            return Err(format!("unaligned tensor {}", r.name));
        }
        if r.dims.contains(&0) {
            return Err(format!("tensor {} has a zero dimension", r.name));
        }
        let elems = r
            .dims
            .iter()
            .try_fold(1u64, |a, &b| a.checked_mul(b))
            .ok_or("tensor element count overflow")?;
        let (block, size) =
            layout(r.dtype).ok_or_else(|| format!("unsupported GGML dtype {}", r.dtype))?;
        if elems % block != 0 {
            return Err(format!("tensor {} does not fill dtype blocks", r.name));
        }
        let len = (elems / block)
            .checked_mul(size)
            .ok_or("tensor byte length overflow")?;
        let offset = base.checked_add(r.offset).ok_or("tensor offset overflow")?;
        if offset.checked_add(len).ok_or("tensor end overflow")? > file_len {
            return Err(format!("tensor {} extends beyond file", r.name));
        }
        out.push(TensorInfo {
            name: r.name,
            dims: r.dims,
            dtype: r.dtype,
            offset,
            len,
        });
    }
    out.sort_by_key(|t| t.offset);
    reject_overlaps(&out)?;
    Ok(out)
}

fn reject_overlaps(tensors: &[TensorInfo]) -> Result<(), String> {
    for pair in tensors.windows(2) {
        let end = pair[0]
            .offset
            .checked_add(pair[0].len)
            .ok_or("tensor end overflow")?;
        if end > pair[1].offset {
            return Err("overlapping tensor ranges".into());
        }
    }
    Ok(())
}

/// Parse and validate a GGUF descriptor without loading tensor bodies.
/// # Errors
/// Returns an error for I/O, unsupported constructs, exceeded limits, invalid arithmetic, or inconsistent ranges.
pub fn read(path: &Path) -> Result<Content, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let file_len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut p = Parser {
        file,
        pos: 0,
        len: file_len,
    };
    let (version, tensor_count, metadata_count) = prelude(&mut p)?;
    let metadata = metadata(&mut p, metadata_count)?;
    let raw = descriptors(&mut p, tensor_count)?;
    let align = metadata
        .get("general.alignment")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32);
    if align == 0 || !align.is_power_of_two() {
        return Err("invalid alignment".into());
    }
    let base = p.pos.checked_add(align - 1).ok_or("alignment overflow")? & !(align - 1);
    if base > file_len {
        return Err("tensor data starts beyond file".into());
    }
    Ok(Content {
        version,
        metadata_count,
        tensor_data_offset: base,
        metadata,
        tensors: ranges(raw, base, file_len, align)?,
    })
}
