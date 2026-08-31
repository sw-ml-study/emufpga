//! The markdown table and the notes that go with it.

use spm_bench::{Backend, BenchRow, Crossover, Sweep};
use std::fmt::Write;

/// Renders a sweep as a markdown table plus what it does and does not
/// support.
#[must_use]
pub fn render(sweep: &Sweep) -> String {
    let mut out = String::new();
    out.push_str("| backend | batch | store MB/s | useful w/s | eta | Ps | Rp | spread |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for row in &sweep.rows {
        out.push_str(&render_row(row));
    }
    out.push('\n');
    out.push_str(&caveats(sweep));
    out
}

/// One measured point as a table row.
fn render_row(row: &BenchRow) -> String {
    let opt = |value: Option<f64>, digits: usize| {
        value.map_or_else(|| "--".to_string(), |v| format!("{v:.digits$}"))
    };
    format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
        row.backend.label(),
        row.batch,
        opt(row.best.raw_bandwidth().map(|b| b / 1.0e6), 1),
        opt(row.best.useful_weights_per_sec().map(|w| w / 1.0e6), 1),
        opt(row.best.eta(), 3),
        opt(row.best.scan_productivity(), 1),
        opt(row.best.residency(), 6),
        opt(row.spread().map(|s| s * 100.0), 1),
    )
}

/// What the table supports, and what it does not.
fn caveats(sweep: &Sweep) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Passes per point: {}. Best pass reported; spread is (slowest - fastest) / fastest.",
        sweep.repeat
    );
    let _ = writeln!(
        out,
        "Timestamp pair overhead on this machine: {:.0} ns, charged to every scale group.",
        sweep.timer_overhead.as_secs_f64() * 1.0e9
    );
    out.push_str(
        "eta = storage_time / compute_time. IO is NOT overlapped in this implementation, so\n\
         the two partition wall clock and eta measures a serial pipeline -- a real result about\n\
         this program, not a prediction of FPGA behaviour, where fetch and compute overlap.\n",
    );
    out.push_str(&crossover_notes(sweep));
    out
}

/// Where each backend sits relative to `eta == 1`.
///
/// The three cases are printed differently on purpose. Saying
/// "crossover at batch 1" when `eta` was already below 1 before the
/// sweep began would report a measurement that was never taken.
fn crossover_notes(sweep: &Sweep) -> String {
    let mut out = String::new();
    for backend in [Backend::Memory, Backend::File] {
        let label = backend.label();
        let _ = match sweep.crossover(backend) {
            Crossover::AlreadyBelow { smallest } => writeln!(
                out,
                "Crossover ({label}): NOT MEASURED -- eta is already below 1 at the smallest\n\
                 batch swept ({smallest}), so the engine is compute-bound across the whole\n\
                 range and the crossing point lies below it."
            ),
            Crossover::At { batch } => writeln!(
                out,
                "Crossover ({label}): eta falls below 1 at batch {batch}."
            ),
            Crossover::NotReached => writeln!(
                out,
                "Crossover ({label}): not reached -- storage remained the limit throughout."
            ),
        };
    }
    out
}
