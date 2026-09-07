mod args;
mod digest;
mod inventory;
mod manifest;
mod profile;
mod verify;

fn main() {
    if let Err(error) = args::run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("spm-model-profile: {error}");
        std::process::exit(1);
    }
}
