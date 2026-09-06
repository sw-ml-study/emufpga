use crate::{manifest, scheduler};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
};

pub fn serve_once(
    socket: &str,
    manifest_path: &str,
    payload: &str,
    clients: usize,
    capacity: usize,
) -> Result<(), String> {
    manifest::verify(manifest_path, payload)?;
    if Path::new(socket).exists() {
        fs::remove_file(socket).map_err(|error| error.to_string())?;
    }
    let listener = UnixListener::bind(socket).map_err(|error| error.to_string())?;
    let mut peers = accept(&listener, clients)?;
    let requests = read_requests(&mut peers)?;
    if requests.iter().any(|id| id != &requests[0]) {
        return Err("batch contains different resources".into());
    }
    let sweep = scheduler::sweep(Path::new(payload), capacity)?;
    for peer in &mut peers {
        writeln!(peer, "OK {} {}", sweep.bytes, sweep.sha256).map_err(|error| error.to_string())?;
    }
    fs::remove_file(socket).map_err(|error| error.to_string())
}

pub fn measurement(
    scope: (&str, usize, usize),
    io: (u64, u64),
    runs: &[scheduler::Sweep],
) -> scheduler::Measurement {
    let (policy, clients, capacity) = scope;
    let (min, max) = runs.iter().fold((f64::INFINITY, 0.0_f64), |range, run| {
        (range.0.min(run.elapsed_ms), range.1.max(run.elapsed_ms))
    });
    scheduler::Measurement {
        policy: policy.into(),
        clients,
        requested_bytes: runs[0].bytes * clients as u64,
        stream_bytes: runs.iter().map(|run| run.bytes).sum(),
        read_bytes: io.1 - io.0,
        buffer_bytes: if policy == "shared" {
            2 * capacity
        } else {
            clients * 1024 * 1024
        },
        elapsed_ms: max,
        fairness: min / max,
        sha256: runs[0].sha256.clone(),
    }
}

fn accept(listener: &UnixListener, clients: usize) -> Result<Vec<UnixStream>, String> {
    (0..clients)
        .map(|_| {
            listener
                .accept()
                .map(|pair| pair.0)
                .map_err(|error| error.to_string())
        })
        .collect()
}

fn read_requests(peers: &mut [UnixStream]) -> Result<Vec<String>, String> {
    peers
        .iter_mut()
        .map(|peer| {
            let mut line = String::new();
            BufReader::new(peer)
                .read_line(&mut line)
                .map_err(|error| error.to_string())?;
            line.strip_prefix("GET ")
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .ok_or("expected GET resource-id".into())
        })
        .collect()
}
