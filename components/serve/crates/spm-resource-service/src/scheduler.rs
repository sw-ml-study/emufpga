use sha2::{Digest, Sha256};
use spm_stream::WeightStream;
use spm_stream_file::PrefetchFileWeightStream;
use std::{path::Path, time::Instant};

pub struct Sweep {
    pub bytes: u64,
    pub sha256: String,
    pub elapsed_ms: f64,
}

pub struct Measurement {
    pub policy: String,
    pub clients: usize,
    pub requested_bytes: u64,
    pub stream_bytes: u64,
    pub read_bytes: u64,
    pub buffer_bytes: usize,
    pub elapsed_ms: f64,
    pub fairness: f64,
    pub sha256: String,
}

pub fn sweep(path: &Path, capacity: usize) -> Result<Sweep, String> {
    let mut stream = PrefetchFileWeightStream::with_capacity(path, capacity)
        .map_err(|error| error.to_string())?;
    let (mut bytes, mut hash) = (0_u64, Sha256::new());
    let mut buffer = vec![0; 64 * 1024];
    let started = Instant::now();
    loop {
        let count = stream
            .next_block(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            return Ok(Sweep {
                bytes,
                sha256: format!("{:x}", hash.finalize()),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            });
        }
        hash.update(&buffer[..count]);
        bytes += count as u64;
    }
}

pub fn measure(
    policy: &str,
    path: &str,
    clients: usize,
    capacity: usize,
) -> Result<Measurement, String> {
    if clients == 0 || capacity == 0 {
        return Err("clients and buffer must be nonzero".into());
    }
    let before = io_bytes()?;
    let runs = match policy {
        "independent" => independent(Path::new(path), clients)?,
        "shared" => vec![sweep(Path::new(path), capacity)?],
        _ => return Err("policy must be independent or shared".into()),
    };
    let after = io_bytes()?;
    Ok(crate::protocol::measurement(
        (policy, clients, capacity),
        (before, after),
        &runs,
    ))
}

fn independent(path: &Path, clients: usize) -> Result<Vec<Sweep>, String> {
    let handles: Vec<_> = (0..clients)
        .map(|_| {
            let path = path.to_owned();
            std::thread::spawn(move || {
                let started = Instant::now();
                let (bytes, sha256) = crate::manifest::payload_digest(&path)?;
                Ok::<_, String>(Sweep {
                    bytes,
                    sha256,
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                })
            })
        })
        .collect();
    handles
        .into_iter()
        .map(|handle| handle.join().map_err(|_| "reader panicked".to_string())?)
        .collect()
}

fn io_bytes() -> Result<u64, String> {
    let text = std::fs::read_to_string("/proc/self/io").map_err(|error| error.to_string())?;
    text.lines()
        .find_map(|line| line.strip_prefix("read_bytes: "))
        .ok_or("missing read_bytes")?
        .parse()
        .map_err(|error| format!("invalid read_bytes: {error}"))
}
