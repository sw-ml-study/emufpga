use spm_file_residency::probe::{Residency, evict, residency};
use std::env;
use std::path::Path;

fn print_json(value: &Residency, path: &Path) {
    let resident_bytes = u64::try_from(value.resident_pages.saturating_mul(value.page_size))
        .unwrap_or(u64::MAX)
        .min(value.bytes);
    println!(
        "{{\"path\":\"{}\",\"device\":{},\"inode\":{},\"bytes\":{},\"page_size\":{},\"pages\":{},\"resident_pages\":{},\"resident_bytes\":{}}}",
        path.display(),
        value.device,
        value.inode,
        value.bytes,
        value.page_size,
        value.pages,
        value.resident_pages,
        resident_bytes
    );
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [command, raw_path] = args.as_slice() else {
        return Err("usage: spm-file-residency <status|evict> FILE".into());
    };
    let path = Path::new(raw_path);
    match command.as_str() {
        "status" => print_json(&residency(path)?, path),
        "evict" => {
            evict(path)?;
            print_json(&residency(path)?, path);
        }
        _ => return Err("usage: spm-file-residency <status|evict> FILE".into()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("spm-file-residency: {error}");
        std::process::exit(1);
    }
}
