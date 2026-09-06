use std::fs::File;
use std::os::fd::AsRawFd;

pub fn page_size() -> Result<usize, String> {
    // SAFETY: sysconf has no pointer arguments or memory-safety preconditions.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(value).map_err(|_| format!("invalid system page size: {value}"))
}

fn map_file(file: &File, length: usize) -> Result<*mut libc::c_void, String> {
    // SAFETY: the descriptor remains open and length is the file length.
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_NONE,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if address == libc::MAP_FAILED {
        Err(format!("mmap: {}", std::io::Error::last_os_error()))
    } else {
        Ok(address)
    }
}

pub fn count_resident_pages(file: &File, length: usize, pages: usize) -> Result<usize, String> {
    let address = map_file(file, length)?;
    let mut vector = vec![0_u8; pages];
    // SAFETY: address is live and vector has one byte for every mapped page.
    let status = unsafe { libc::mincore(address, length, vector.as_mut_ptr()) };
    let saved_error = std::io::Error::last_os_error();
    // SAFETY: address and length describe the mapping created above.
    let unmap_status = unsafe { libc::munmap(address, length) };
    if status != 0 {
        return Err(format!("mincore: {saved_error}"));
    }
    if unmap_status != 0 {
        return Err(format!("munmap: {}", std::io::Error::last_os_error()));
    }
    Ok(vector.iter().filter(|value| **value & 1 == 1).count())
}
