use crate::{MAX_ARRAY, MAX_HEADER_BYTES, MAX_STRING};
use std::{fs::File, io::Read};

pub(crate) struct Parser {
    pub file: File,
    pub pos: u64,
    pub len: u64,
}

impl Parser {
    pub fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let end = self
            .pos
            .checked_add(n as u64)
            .ok_or("file position overflow")?;
        if end > MAX_HEADER_BYTES {
            return Err("GGUF header exceeds limit".into());
        }
        if end > self.len {
            return Err("unexpected end of GGUF".into());
        }
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| "allocation refused")?;
        out.resize(n, 0);
        self.file.read_exact(&mut out).map_err(|e| e.to_string())?;
        self.pos = end;
        Ok(out)
    }
    pub fn u8(&mut self) -> Result<u8, String> {
        Ok(self.bytes(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    pub fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return Err(format!("string length {n} exceeds limit"));
        }
        String::from_utf8(self.bytes(usize::try_from(n).map_err(|_| "string too large")?)?)
            .map_err(|_| "invalid UTF-8 string".into())
    }
    pub fn value(&mut self, ty: u32, depth: u8) -> Result<Option<String>, String> {
        if depth > 4 {
            return Err("metadata nesting exceeds limit".into());
        }
        Ok(match ty {
            0 => Some(self.u8()?.to_string()),
            1 => Some(i8::from_le_bytes([self.u8()?]).to_string()),
            2 => Some(self.u16()?.to_string()),
            3 => Some(i16::from_le_bytes(self.u16()?.to_le_bytes()).to_string()),
            4 => Some(self.u32()?.to_string()),
            5 => Some(i32::from_le_bytes(self.u32()?.to_le_bytes()).to_string()),
            6 => {
                self.bytes(4)?;
                None
            }
            7 => {
                let v = self.u8()?;
                if v > 1 {
                    return Err("invalid boolean metadata value".into());
                }
                Some(v.to_string())
            }
            8 => Some(self.string()?),
            9 => {
                let inner = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARRAY {
                    return Err(format!("array length {n} exceeds limit"));
                }
                for _ in 0..n {
                    self.value(inner, depth + 1)?;
                }
                None
            }
            10 => Some(self.u64()?.to_string()),
            11 => Some(i64::from_le_bytes(self.u64()?.to_le_bytes()).to_string()),
            12 => {
                self.bytes(8)?;
                None
            }
            _ => return Err(format!("unknown metadata type {ty}")),
        })
    }
}
