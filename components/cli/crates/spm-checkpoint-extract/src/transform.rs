//! Numeric conversion and physical layout transformation.

use half::f16;
use spm_checkpoint_source::DType;

pub(crate) fn convert(raw: &[u8], dtype: DType, shape: &[usize], bf16: bool) -> Vec<u8> {
    let widened = widen(raw, dtype);
    let ordered = transpose(&widened, shape);
    if bf16 { narrow_bf16(&ordered) } else { ordered }
}

fn widen(raw: &[u8], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => raw.to_vec(),
        DType::Bf16 => raw
            .chunks_exact(2)
            .flat_map(|v| { u32::from(u16::from_le_bytes([v[0], v[1]])) << 16 }.to_le_bytes())
            .collect(),
        DType::F16 => raw
            .chunks_exact(2)
            .flat_map(|v| {
                f16::from_bits(u16::from_le_bytes([v[0], v[1]]))
                    .to_f32()
                    .to_le_bytes()
            })
            .collect(),
    }
}

fn transpose(raw: &[u8], shape: &[usize]) -> Vec<u8> {
    if shape.len() < 2 {
        return raw.to_vec();
    }
    let cols = *shape.last().unwrap_or(&1);
    let rows = shape[..shape.len() - 1].iter().product::<usize>();
    let mut out = vec![0; raw.len()];
    for column in 0..cols {
        for row in 0..rows {
            let src = (row * cols + column) * 4;
            let dst = (column * rows + row) * 4;
            out[dst..dst + 4].copy_from_slice(&raw[src..src + 4]);
        }
    }
    out
}

fn narrow_bf16(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(4)
        .flat_map(|v| {
            let bits = u32::from_le_bytes([v[0], v[1], v[2], v[3]]);
            let nan = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
            let rounded = if nan {
                (bits >> 16) | 0x40
            } else {
                (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
            };
            u16::try_from(rounded & 0xffff)
                .unwrap_or_default()
                .to_le_bytes()
        })
        .collect()
}
