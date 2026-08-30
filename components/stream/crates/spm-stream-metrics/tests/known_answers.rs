//! Every metric checked against a value computed by hand in the test.
//!
//! These are the numbers the whole project is judged on, so none of
//! them is asserted against "whatever the code produces".

use spm_stream_metrics::ScanMetrics;
use std::time::Duration;

/// A scan of a 1000-byte model: 4000 ternary weights at 2 bits each,
/// read once, applied once per weight (batch 1). Storage took 2 s,
/// compute took 1 s. One 16-byte group is resident.
fn batch_one() -> ScanMetrics {
    ScanMetrics {
        parameter_bytes_read: 1000,
        weights_decoded: 4000,
        weight_applications: 4000,
        resident_parameter_bytes: 16,
        total_parameter_bytes: 1000,
        storage_time: Duration::from_secs(2),
        compute_time: Duration::from_secs(1),
    }
}

#[test]
fn batch_one_gives_a_scan_productivity_of_exactly_one() {
    // Ps = applications / weights read = 4000 / 4000. Batch-1 dense
    // inference reads every weight, uses it once, throws it away.
    // If this is ever not 1.0, the accounting is wrong.
    assert_eq!(batch_one().scan_productivity(), Some(1.0));
}

#[test]
fn batching_multiplies_scan_productivity_by_the_batch_size() {
    // Same scan, batch 32: each decoded weight is applied to 32
    // activations, so 4000 * 32 = 128_000 applications over the same
    // 4000 weights read. Ps = 32. This is the architecture's entire
    // economic argument in one assertion.
    let metrics = ScanMetrics {
        weight_applications: 4000 * 32,
        ..batch_one()
    };
    assert_eq!(metrics.scan_productivity(), Some(32.0));
    // Storage did not get any busier.
    assert_eq!(metrics.raw_bandwidth(), batch_one().raw_bandwidth());
}

#[test]
fn residency_reports_the_buffer_rather_than_rounding_it_away() {
    // Rp = 16 resident bytes / 1000 total = 0.016. A group-at-a-time
    // streamer is never at zero, and claiming zero would overstate
    // the result.
    assert_eq!(batch_one().residency(), Some(0.016));
    // Conventional inference loads the model: Rp = 1.
    let loaded = ScanMetrics {
        resident_parameter_bytes: 1000,
        ..batch_one()
    };
    assert_eq!(loaded.residency(), Some(1.0));
}

#[test]
fn eta_says_whether_compute_keeps_up_with_storage() {
    // eta = storage_time / compute_time = 2 / 1 = 2. Compute is twice
    // as fast as it needs to be, so the scan is storage-bound -- the
    // regime the architecture is designed for.
    assert_eq!(batch_one().eta(), Some(2.0));

    // Flip it: storage 1 s, compute 4 s -> eta = 0.25, compute-bound,
    // and the hardware target is "4x faster engine, or 4x the lanes".
    let compute_bound = ScanMetrics {
        storage_time: Duration::from_secs(1),
        compute_time: Duration::from_secs(4),
        ..batch_one()
    };
    assert_eq!(compute_bound.eta(), Some(0.25));
}
