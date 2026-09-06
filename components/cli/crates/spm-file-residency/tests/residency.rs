use spm_file_residency::probe::residency;
use std::fs::File;
use std::io::Write;

#[test]
fn reports_a_small_file_without_changing_it() {
    let path = std::env::temp_dir().join(format!("spm-residency-{}", std::process::id()));
    let mut file = File::create(&path).expect("create fixture");
    file.write_all(&[42_u8; 8192]).expect("write fixture");
    file.sync_all().expect("sync fixture");
    let result = residency(&path).expect("query residency");
    assert_eq!(result.bytes, 8192);
    assert_eq!(result.pages, 8192_usize.div_ceil(result.page_size));
    assert!(result.resident_pages <= result.pages);
    std::fs::remove_file(path).expect("remove fixture");
}
