use std::{env, path::Path};

fn main() {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: spm-gguf-inspect <model.gguf>");
        std::process::exit(2);
    };
    match spm_gguf::read(Path::new(&path)) {
        Ok(c) => {
            println!(
                "version={} metadata={} tensors={} data_offset={}",
                c.version,
                c.metadata_count,
                c.tensors.len(),
                c.tensor_data_offset
            );
            for key in [
                "general.architecture",
                "qwen3.context_length",
                "general.alignment",
            ] {
                if let Some(v) = c.metadata.get(key) {
                    println!("{key}={v}");
                }
            }
            for t in &c.tensors {
                println!(
                    "tensor\t{}\t{:?}\ttype={}\toffset={}\tbytes={}",
                    t.name, t.dims, t.dtype, t.offset, t.len
                );
            }
        }
        Err(e) => {
            eprintln!("spm-gguf-inspect: {e}");
            std::process::exit(1);
        }
    }
}
