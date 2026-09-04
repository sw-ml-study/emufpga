use spm_stream::StreamError;
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
    sync::mpsc::{self, Receiver},
    thread::{self, JoinHandle},
};

pub(crate) type Chunk = io::Result<Vec<u8>>;

pub(crate) fn spawn_reader(
    path: &Path,
    capacity: usize,
) -> Result<(Receiver<Chunk>, JoinHandle<()>), StreamError> {
    let mut file = File::open(path)?;
    let (sender, receiver) = mpsc::sync_channel(0);
    let worker = thread::spawn(move || {
        loop {
            let mut chunk = vec![0; capacity];
            let mut filled = 0;
            while filled < capacity {
                match file.read(&mut chunk[filled..]) {
                    Ok(0) => break,
                    Ok(count) => filled += count,
                    Err(error) => {
                        sender.send(Err(error)).ok();
                        return;
                    }
                }
            }
            chunk.truncate(filled);
            let done = chunk.is_empty();
            if sender.send(Ok(chunk)).is_err() || done {
                return;
            }
        }
    });
    Ok((receiver, worker))
}
