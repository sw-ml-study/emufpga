use std::{env, fs, path::PathBuf, time::Duration};

#[derive(Clone, Copy)]
pub enum Backend {
    Sync,
    Prefetch,
}

pub struct Args {
    pub path: PathBuf,
    pub backend: Backend,
    pub capacity: usize,
}

pub fn parse() -> Result<Args, String> {
    let values: Vec<_> = env::args().skip(1).collect();
    if values.len() != 3 {
        return Err("usage: spm-stream-replay FILE sync|prefetch BUFFER_BYTES".into());
    }
    let backend = match values[1].as_str() {
        "sync" => Backend::Sync,
        "prefetch" => Backend::Prefetch,
        other => return Err(format!("unknown backend: {other}")),
    };
    let capacity = values[2].parse().map_err(|_| "invalid buffer size")?;
    Ok(Args {
        path: values[0].clone().into(),
        backend,
        capacity,
    })
}

pub struct Io {
    pub read_bytes: u64,
    pub rchar: u64,
}

pub fn io() -> Result<Io, String> {
    let text = fs::read_to_string("/proc/self/io").map_err(|error| error.to_string())?;
    Ok(Io {
        read_bytes: field(&text, "read_bytes")?,
        rchar: field(&text, "rchar")?,
    })
}

fn field(text: &str, name: &str) -> Result<u64, String> {
    text.lines()
        .find_map(|line| line.strip_prefix(&format!("{name}: ")))
        .ok_or_else(|| format!("missing /proc/self/io field: {name}"))?
        .parse()
        .map_err(|error| format!("invalid {name}: {error}"))
}

pub fn print(bytes: u64, digest: u64, elapsed: Duration, before: &Io, after: &Io, resident: usize) {
    let seconds = elapsed.as_secs_f64();
    let kib = u32::try_from(bytes / 1024).map_or(f64::from(u32::MAX), f64::from);
    println!(
        "{{\"schema\":\"emufpga.stream-replay.v1\",\"bytes\":{bytes},\"digest\":\"{digest:016x}\",\"elapsed_ms\":{:.3},\"bandwidth_mib_s\":{:.3},\"read_bytes\":{},\"rchar\":{},\"buffer_resident_bytes\":{resident}}}",
        seconds * 1000.0,
        kib / seconds / 1024.0,
        after.read_bytes - before.read_bytes,
        after.rchar - before.rchar
    );
}
