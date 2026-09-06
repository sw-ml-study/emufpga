//! Physical trace replay for an execution-order parameter stream.

mod digest;
mod report;
mod run;

fn main() {
    if let Err(error) = run::execute() {
        eprintln!("spm-stream-replay: {error}");
        std::process::exit(1);
    }
}
