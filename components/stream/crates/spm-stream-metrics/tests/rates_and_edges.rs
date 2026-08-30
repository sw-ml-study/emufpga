//! Throughput rates, and the zero-denominator cases that must report
//! "unknown" rather than a plausible zero.

use spm_stream_metrics::ScanMetrics;
use std::time::Duration;

#[test]
fn bandwidth_is_measured_against_storage_time_alone() {
    // 1000 bytes in 2 s of storage time = 500 B/s, regardless of how
    // long compute took. This reports what the store can deliver, not
    // what the pipeline achieved.
    let metrics = ScanMetrics {
        parameter_bytes_read: 1000,
        storage_time: Duration::from_secs(2),
        compute_time: Duration::from_secs(8),
        ..ScanMetrics::default()
    };
    assert_eq!(metrics.raw_bandwidth(), Some(500.0));
}

#[test]
fn weight_rates_are_measured_against_wall_clock() {
    // 4000 weights over 2 s storage + 2 s compute = 4 s wall clock
    // -> 1000 weights/s decoded. With batch 8 the useful rate is 8x
    // that, off the same 4 s.
    let metrics = ScanMetrics {
        weights_decoded: 4000,
        weight_applications: 32_000,
        storage_time: Duration::from_secs(2),
        compute_time: Duration::from_secs(2),
        ..ScanMetrics::default()
    };
    assert_eq!(metrics.elapsed(), Duration::from_secs(4));
    assert_eq!(metrics.decoded_weights_per_sec(), Some(1000.0));
    assert_eq!(metrics.useful_weights_per_sec(), Some(8000.0));
}

#[test]
fn zero_denominators_report_unknown_not_zero() {
    // A scan that decoded no weights has no scan productivity. That
    // is not the same as a scan productivity of zero, and reporting
    // 0.0 would put a false data point on a chart.
    let empty = ScanMetrics::default();
    assert_eq!(empty.scan_productivity(), None);
    assert_eq!(empty.residency(), None);
    assert_eq!(empty.eta(), None);
    assert_eq!(empty.raw_bandwidth(), None);
    assert_eq!(empty.decoded_weights_per_sec(), None);
}

#[test]
fn counters_stay_exact_above_the_f64_integer_range() {
    // Byte counts for a 70B model exceed 2^32, so the widening path
    // has to be right. 2^40 bytes read in 1 s.
    let metrics = ScanMetrics {
        parameter_bytes_read: 1 << 40,
        weights_decoded: 1 << 40,
        weight_applications: 1 << 42,
        storage_time: Duration::from_secs(1),
        compute_time: Duration::from_secs(1),
        ..ScanMetrics::default()
    };
    assert_eq!(metrics.raw_bandwidth(), Some(1_099_511_627_776.0));
    // 2^42 / 2^40 = 4 exactly.
    assert_eq!(metrics.scan_productivity(), Some(4.0));
}
