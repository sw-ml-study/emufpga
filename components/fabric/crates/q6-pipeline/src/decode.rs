pub(crate) fn decode_block(block: &[u8], out: &mut Vec<f32>) {
    let (ql, rest) = block.split_at(128);
    let (qh, rest) = rest.split_at(64);
    let (scales, d) = rest.split_at(16);
    let delta = f16(u16::from_le_bytes([d[0], d[1]]));
    for half in 0..2 {
        let base = out.len();
        out.resize(base + 128, 0.0);
        for lane in 0..32 {
            let s = lane / 16;
            let low = &ql[half * 64..];
            let high = &qh[half * 32..];
            let q = [
                (low[lane] & 15) | ((high[lane] & 3) << 4),
                (low[lane + 32] & 15) | (((high[lane] >> 2) & 3) << 4),
                (low[lane] >> 4) | (((high[lane] >> 4) & 3) << 4),
                (low[lane + 32] >> 4) | (((high[lane] >> 6) & 3) << 4),
            ];
            for group in 0..4 {
                out[base + lane + group * 32] = delta
                    * f32::from(i8::from_ne_bytes([scales[half * 8 + s + group * 2]]))
                    * (f32::from(q[group]) - 32.0);
            }
        }
    }
}
fn f16(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exp = u32::from((value >> 10) & 31);
    let frac = u32::from(value & 1023);
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut f = frac;
            let mut e = -14i32;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (u32::try_from(e + 127).unwrap_or_default() << 23) | ((f & 1023) << 13)
        }
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}
