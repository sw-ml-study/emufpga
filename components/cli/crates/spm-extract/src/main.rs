//! Rust-native checkpoint extraction entry point.

use spm_checkpoint_extract::{Encoding, extract};
use std::{env, path::Path};

fn main() {
    if let Err(error) = run() {
        eprintln!("spm-extract: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let encoding = if args.first().is_some_and(|arg| arg == "--bf16") {
        args.remove(0);
        Encoding::Bf16
    } else {
        Encoding::F32
    };
    let [source, output] = args.as_slice() else {
        return Err("usage: spm-extract [--bf16] <model.pt|.safetensors> <out-dir>".into());
    };
    let summary = extract(Path::new(source), Path::new(output), encoding)?;
    let dtype = if encoding == Encoding::Bf16 {
        "bf16"
    } else {
        "f32"
    };
    println!(
        "{} tensors, {} parameters, {} bytes ({dtype}) -> {output}",
        summary.tensors, summary.parameters, summary.bytes
    );
    Ok(())
}
