use crate::mapping::{count_resident_pages, page_size};
use std::fs::{File, Metadata};
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

#[derive(Debug)]
pub struct Residency {
    pub bytes: u64,
    pub pages: usize,
    pub resident_pages: usize,
    pub page_size: usize,
    pub device: u64,
    pub inode: u64,
}

fn open(path: &Path) -> Result<File, String> {
    File::open(path).map_err(|error| format!("open {}: {error}", path.display()))
}

fn report(metadata: &Metadata, pages: usize, resident_pages: usize, size: usize) -> Residency {
    Residency {
        bytes: metadata.len(),
        pages,
        resident_pages,
        page_size: size,
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

/// Reports the file's page-cache residency without reading its contents.
///
/// # Errors
///
/// Returns an error when file metadata, address-space mapping, or `mincore` fails.
pub fn residency(path: &Path) -> Result<Residency, String> {
    let file = open(path)?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?;
    let bytes = metadata.len();
    let size = page_size()?;
    let pages =
        usize::try_from(bytes.div_ceil(size as u64)).map_err(|_| "file has too many pages")?;
    let resident_pages = if pages == 0 {
        0
    } else {
        let length =
            usize::try_from(bytes).map_err(|_| "file is too large for this address space")?;
        count_resident_pages(&file, length, pages)
            .map_err(|error| format!("{}: {error}", path.display()))?
    };
    Ok(report(&metadata, pages, resident_pages, size))
}

/// Advises Linux that every clean cached page of the file may be discarded.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or `posix_fadvise` fails.
pub fn evict(path: &Path) -> Result<(), String> {
    let file = open(path)?;
    // SAFETY: fd refers to an open regular file; the range covers the file.
    let status = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "posix_fadvise {}: {}",
            path.display(),
            std::io::Error::from_raw_os_error(status)
        ))
    }
}
