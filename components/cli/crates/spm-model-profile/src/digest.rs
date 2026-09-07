use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub fn file(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
