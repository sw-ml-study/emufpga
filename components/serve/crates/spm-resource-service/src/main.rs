//! Minimal Linux model-resource service and benchmark.

use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::UnixStream,
};

mod manifest;
mod protocol;
mod scheduler;

fn main() {
    if let Err(error) = run() {
        eprintln!("spm-resource-service: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |value: &str| {
        value
            .parse()
            .map_err(|_| format!("invalid integer: {value}"))
    };
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["verify", manifest, payload] => manifest::verify(manifest, payload),
        ["measure", policy, payload, clients, buffer] => {
            let run = scheduler::measure(policy, payload, number(clients)?, number(buffer)?)?;
            print_measurement(&run);
            Ok(())
        }
        ["serve-once", socket, manifest, payload, clients, buffer] => {
            protocol::serve_once(socket, manifest, payload, number(clients)?, number(buffer)?)
        }
        ["request", socket, id] => request(socket, id),
        _ => Err("usage: spm-resource-service verify MANIFEST PAYLOAD | measure independent|shared PAYLOAD CLIENTS BUFFER_BYTES | serve-once SOCKET MANIFEST PAYLOAD CLIENTS BUFFER_BYTES | request SOCKET RESOURCE_ID".into()),
    }
}

fn print_measurement(run: &scheduler::Measurement) {
    println!(
        "{{\"schema\":\"emufpga.resource-measurement.v1\",\"policy\":\"{}\",\"clients\":{},\"requested_bytes\":{},\"stream_bytes\":{},\"read_bytes\":{},\"buffer_bytes\":{},\"elapsed_ms\":{:.3},\"fairness_ratio\":{:.6},\"sha256\":\"{}\"}}",
        run.policy,
        run.clients,
        run.requested_bytes,
        run.stream_bytes,
        run.read_bytes,
        run.buffer_bytes,
        run.elapsed_ms,
        run.fairness,
        run.sha256
    );
}

fn request(socket: &str, id: &str) -> Result<(), String> {
    let mut peer = UnixStream::connect(socket).map_err(|error| error.to_string())?;
    writeln!(peer, "GET {id}").map_err(|error| error.to_string())?;
    let mut reply = String::new();
    BufReader::new(peer)
        .read_line(&mut reply)
        .map_err(|error| error.to_string())?;
    println!("{}", reply.trim());
    Ok(())
}
