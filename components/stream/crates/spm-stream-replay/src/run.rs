use crate::{digest::Digest, report};
use spm_stream::WeightStream;
use spm_stream_file::{FileWeightStream, PrefetchFileWeightStream};
use std::time::Instant;

pub fn execute() -> Result<(), String> {
    let args = report::parse()?;
    let mut stream: Box<dyn WeightStream> = match args.backend {
        report::Backend::Sync => Box::new(
            FileWeightStream::with_capacity(&args.path, args.capacity)
                .map_err(|error| error.to_string())?,
        ),
        report::Backend::Prefetch => Box::new(
            PrefetchFileWeightStream::with_capacity(&args.path, args.capacity)
                .map_err(|error| error.to_string())?,
        ),
    };
    replay(&mut stream, args.capacity)
}

fn replay(stream: &mut dyn WeightStream, capacity: usize) -> Result<(), String> {
    let before = report::io()?;
    let started = Instant::now();
    let (bytes, digest) = drain(stream)?;
    let elapsed = started.elapsed();
    let after = report::io()?;
    report::print(bytes, digest, elapsed, &before, &after, 2 * capacity);
    Ok(())
}

fn drain(stream: &mut dyn WeightStream) -> Result<(u64, u64), String> {
    let mut buffer = vec![0; 64 * 1024];
    let (mut bytes, mut digest) = (0_u64, Digest::new());
    loop {
        let taken = stream
            .next_block(&mut buffer)
            .map_err(|error| error.to_string())?;
        if taken == 0 {
            return Ok((bytes, digest.value()));
        }
        digest.update(&buffer[..taken]);
        bytes += taken as u64;
    }
}
