//! Accumulator banks: add, subtract, and lane independence.

use spm_accum::AccumulatorBank;

#[test]
fn add_and_subtract_are_the_only_operations() {
    let mut bank = AccumulatorBank::new(1, 3);
    bank.accumulate(0, false, &[2.5]);
    bank.accumulate(0, false, &[2.5]);
    bank.accumulate(1, true, &[4.0]);
    assert_eq!(bank.lane(0), &[5.0, -4.0, 0.0]);
}

#[test]
fn one_weight_reaches_every_lane() {
    // This is the reuse the architecture is buying: the weight was
    // read once and applied four times.
    let mut bank = AccumulatorBank::new(4, 2);
    bank.accumulate(1, false, &[1.0, 2.0, 3.0, 4.0]);
    for (lane, expected) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
        assert_eq!(bank.lane(lane), &[0.0, expected]);
    }
}

#[test]
fn lanes_do_not_bleed_into_each_other() {
    let mut bank = AccumulatorBank::new(3, 4);
    bank.accumulate(3, false, &[1.0, 0.0, 0.0]);
    assert_eq!(bank.lane(0), &[0.0, 0.0, 0.0, 1.0]);
    assert_eq!(bank.lane(1), &[0.0; 4]);
    assert_eq!(bank.lane(2), &[0.0; 4]);
}

#[test]
fn reset_clears_every_lane() {
    let mut bank = AccumulatorBank::new(2, 2);
    bank.accumulate(0, false, &[1.0, 1.0]);
    bank.reset();
    assert_eq!(bank.lane(0), &[0.0, 0.0]);
    assert_eq!(bank.lane(1), &[0.0, 0.0]);
}
