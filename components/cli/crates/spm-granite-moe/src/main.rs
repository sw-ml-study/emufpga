use std::{env, path::Path};

mod attention;
mod layout;
mod math;
mod model;
mod moe;
mod shape;
mod smoke;
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
    if args.len() == 5 && args[1] == "--expert-smoke" {
        let batch = args[4]
            .parse()
            .map_err(|_| "invalid smoke batch".to_owned());
        let result =
            batch.and_then(|batch| smoke::run(Path::new(&args[2]), Path::new(&args[3]), batch));
        if let Err(error) = result {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.len() != 3 {
        eprintln!(
            "usage: spm-granite-moe MODEL.gguf TOKEN_ID[,TOKEN_ID...]\n       spm-granite-moe --expert-smoke MODEL.gguf OUTPUT.spm BATCH"
        );
        std::process::exit(2);
    }
    let result = tokens(&args[2]).and_then(|ids| model::run(Path::new(&args[1]), &ids));
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
