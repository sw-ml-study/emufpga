use std::{env, path::Path};

mod attention;
mod compare;
mod math;
mod model;
mod weights;

fn parse_tokens(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|token| {
            token
                .parse()
                .map_err(|_| format!("invalid token ID {token}"))
        })
        .collect()
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: spm-qwen3-slice MODEL.gguf TOKEN_ID[,TOKEN_ID...]");
        std::process::exit(2);
    }
    let result = parse_tokens(&args[2]).and_then(|tokens| model::run(Path::new(&args[1]), &tokens));
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
