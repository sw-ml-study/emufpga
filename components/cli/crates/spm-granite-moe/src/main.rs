use std::{env, path::Path};

mod attention;
mod math;
mod model;
mod moe;
mod weights;

fn tokens(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|value| {
            value
                .parse()
                .map_err(|_| format!("invalid token ID {value}"))
        })
        .collect()
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: spm-granite-moe MODEL.gguf TOKEN_ID[,TOKEN_ID...]");
        std::process::exit(2);
    }
    let result = tokens(&args[2]).and_then(|ids| model::run(Path::new(&args[1]), &ids));
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
