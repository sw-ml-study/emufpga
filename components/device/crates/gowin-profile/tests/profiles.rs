//! Every figure is either sourced or explicitly unknown, and the
//! primary board's fabric is complete.

use gowin_profile::{Figure, NANO_9K, NANO_20K, all};

#[test]
fn every_known_figure_carries_a_source() {
    // The point of the type: a number without provenance should not
    // be expressible, and this checks none slipped through with an
    // empty citation.
    for profile in all() {
        for (name, figure) in [
            ("lut4", profile.lut4),
            ("flip_flops", profile.flip_flops),
            ("bsram_bits", profile.bsram_bits),
            ("dsp_18x18", profile.dsp_18x18),
            ("bulk.bits", profile.bulk.bits),
        ] {
            if let Figure::Known { source, .. } = figure {
                assert!(
                    !source.document.is_empty() && !source.url.is_empty(),
                    "{} {name}: known figure without a citation",
                    profile.board
                );
                assert!(
                    source.url.starts_with("https://"),
                    "{} {name}: citation is not a URL",
                    profile.board
                );
            }
        }
    }
}

#[test]
fn the_primary_boards_fabric_is_complete() {
    // The 9K is the primary target. A fit model cannot place an engine
    // on it without these four, so a gap here blocks step 008.
    assert!(
        NANO_9K.fabric_is_complete(),
        "9K fabric incomplete: {:?}",
        NANO_9K.unknown_fields()
    );
    assert_eq!(NANO_9K.lut4.value(), Some(8640));
    assert_eq!(NANO_9K.flip_flops.value(), Some(6480));
    assert_eq!(NANO_9K.bsram_bits.value(), Some(468 * 1024));
    assert_eq!(NANO_9K.dsp_18x18.value(), Some(20));
}

#[test]
fn bandwidth_and_fmax_are_unknown_everywhere() {
    // Not an oversight -- the two gaps that matter, recorded so step
    // 008 cannot quietly assume values for them. If a source is ever
    // found, this test is what tells you to revisit the fit model.
    for profile in all() {
        assert!(
            !profile.bulk.bandwidth_mbps.is_known(),
            "{}: bulk bandwidth became known -- revisit docs/fit-model.md",
            profile.board
        );
        assert!(
            !profile.fmax_mhz.is_known(),
            "{}: fmax became known -- revisit docs/fit-model.md",
            profile.board
        );
    }
}

#[test]
fn an_unknown_cannot_be_read_as_a_number() {
    // Figure::value() returns Option, so there is no path that turns
    // a missing LUT4 count into a zero.
    let unknown = Figure::Unknown { note: "test" };
    assert_eq!(unknown.value(), None);
    assert_eq!(unknown.source(), None);
    assert!(!unknown.is_known());
    assert!(format!("{unknown}").contains("unknown"));
}

#[test]
fn larger_parts_do_not_report_fewer_luts() {
    // Ordering sanity: a transcription error that swapped two boards
    // would most likely show up here.
    let known: Vec<(&str, u64)> = all()
        .iter()
        .filter_map(|p| p.lut4.value().map(|v| (p.board, v)))
        .collect();
    for pair in known.windows(2) {
        assert!(
            pair[0].1 <= pair[1].1,
            "{} ({}) reports more LUT4 than {} ({})",
            pair[0].0,
            pair[0].1,
            pair[1].0,
            pair[1].1
        );
    }
}

#[test]
fn the_20k_has_the_widest_documented_bulk_interface() {
    // The only board whose bulk memory width is stated at all, and
    // the reason it is the interesting second target: 32-bit SDRAM
    // against everything else's unstated PSRAM.
    assert_eq!(NANO_20K.bulk.width_bits.value(), Some(32));
    assert_eq!(NANO_20K.bulk.kind, "SDR SDRAM");
    for profile in all() {
        if profile.board != NANO_20K.board {
            assert!(
                !profile.bulk.width_bits.is_known(),
                "{}: bulk width became known",
                profile.board
            );
        }
    }
}
