use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Serialize)]
pub struct RankedExpert {
    pub resource: String,
    pub assignments: u64,
    pub trace_presence: usize,
}

#[derive(Serialize)]
pub struct Coactivation {
    pub resources: [String; 2],
    pub events: u64,
}

#[derive(Serialize)]
pub struct WorkloadProfile {
    pub state: String,
    pub trace_sha256: Vec<String>,
    pub events: u64,
    pub assignments: u64,
    pub unique_applications: u64,
    pub reuse_ppm: u64,
    pub heldout_top32_overlap_ppm: Option<u64>,
    pub top_coactivations: Vec<Coactivation>,
    pub ranking: Vec<RankedExpert>,
}

pub fn build(paths: &[PathBuf]) -> Result<WorkloadProfile, String> {
    if paths.is_empty() {
        return Ok(cold());
    }
    let (counts, per_trace, coactivations, hashes, events, unique_applications) = aggregate(paths)?;
    let assignments = counts.values().sum();
    let mut ranking: Vec<_> = counts
        .into_iter()
        .map(|(resource, assignments)| RankedExpert {
            trace_presence: per_trace
                .iter()
                .filter(|trace| trace.contains_key(&resource))
                .count(),
            resource,
            assignments,
        })
        .collect();
    ranking.sort_by(|left, right| {
        right
            .assignments
            .cmp(&left.assignments)
            .then(left.resource.cmp(&right.resource))
    });
    let mut top_coactivations: Vec<_> = coactivations
        .into_iter()
        .map(|(resources, events)| Coactivation { resources, events })
        .collect();
    top_coactivations.sort_by(|a, b| b.events.cmp(&a.events).then(a.resources.cmp(&b.resources)));
    top_coactivations.truncate(64);
    Ok(WorkloadProfile {
        state: "calibrated".into(),
        trace_sha256: hashes,
        events,
        assignments,
        unique_applications,
        reuse_ppm: ratio_ppm(assignments.saturating_sub(unique_applications), assignments),
        heldout_top32_overlap_ppm: overlap(&per_trace),
        top_coactivations,
        ranking,
    })
}

fn cold() -> WorkloadProfile {
    WorkloadProfile {
        state: "cold_fallback".into(),
        trace_sha256: Vec::new(),
        events: 0,
        assignments: 0,
        unique_applications: 0,
        reuse_ppm: 0,
        heldout_top32_overlap_ppm: None,
        top_coactivations: Vec::new(),
        ranking: Vec::new(),
    }
}

type Aggregate = (
    BTreeMap<String, u64>,
    Vec<BTreeMap<String, u64>>,
    BTreeMap<[String; 2], u64>,
    Vec<String>,
    u64,
    u64,
);

fn aggregate(paths: &[PathBuf]) -> Result<Aggregate, String> {
    let (mut counts, mut per_trace, mut pairs, mut hashes) =
        (BTreeMap::new(), Vec::new(), BTreeMap::new(), Vec::new());
    let (mut events, mut unique) = (0, 0);
    for path in paths {
        let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let mut trace = BTreeMap::new();
        hashes.push(crate::digest::bytes(text.as_bytes()));
        collect(&text, &mut trace, &mut pairs, &mut events, &mut unique)?;
        for (resource, count) in &trace {
            *counts.entry(resource.clone()).or_default() += count;
        }
        per_trace.push(trace);
    }
    Ok((counts, per_trace, pairs, hashes, events, unique))
}

fn collect(
    text: &str,
    counts: &mut BTreeMap<String, u64>,
    coactivations: &mut BTreeMap<[String; 2], u64>,
    events: &mut u64,
    unique: &mut u64,
) -> Result<(), String> {
    for line in text.lines().filter(|line| {
        line.starts_with("spm_mmid_experts ") && line.contains("ffn_gate_up_exps.weight")
    }) {
        let layer = line
            .split("tensor=blk.")
            .nth(1)
            .and_then(|part| part.split('.').next())
            .ok_or("bad trace layer")?;
        let pairs = line.split(" counts=").nth(1).ok_or("bad trace counts")?;
        let mut active = Vec::new();
        for pair in pairs.split(',') {
            let (expert, count) = pair.split_once(':').ok_or("bad expert count")?;
            let resource = format!("{layer}:{expert}");
            *counts.entry(resource.clone()).or_default() +=
                count.parse::<u64>().map_err(|error| error.to_string())?;
            active.push(resource);
            *unique += 1;
        }
        for left in 0..active.len() {
            for right in left + 1..active.len() {
                *coactivations
                    .entry([active[left].clone(), active[right].clone()])
                    .or_default() += 1;
            }
        }
        *events += 1;
    }
    Ok(())
}

fn ratio_ppm(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(1_000_000) / denominator
    }
}

fn overlap(traces: &[BTreeMap<String, u64>]) -> Option<u64> {
    let (heldout, calibration) = traces.split_last()?;
    if calibration.is_empty() {
        return None;
    }
    let mut combined = BTreeMap::new();
    for trace in calibration {
        for (key, count) in trace {
            *combined.entry(key.clone()).or_insert(0_u64) += count;
        }
    }
    let top = |counts: &BTreeMap<String, u64>| {
        let mut rows: Vec<_> = counts.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        rows.into_iter()
            .take(32)
            .map(|(key, _)| key.clone())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let expected = top(&combined);
    let observed = top(heldout);
    Some(ratio_ppm(
        expected.intersection(&observed).count() as u64,
        expected.len() as u64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_reuse_and_heldout_overlap() {
        let line = "spm_mmid_experts tensor=blk.2.ffn_gate_up_exps.weight counts=7:2,9:1\n";
        let mut counts = BTreeMap::new();
        let mut coactivations = BTreeMap::new();
        let (mut events, mut unique) = (0, 0);
        collect(
            line,
            &mut counts,
            &mut coactivations,
            &mut events,
            &mut unique,
        )
        .unwrap();
        assert_eq!((events, unique, counts["2:7"]), (1, 2, 2));
        assert_eq!(coactivations[&["2:7".into(), "2:9".into()]], 1);
        assert_eq!(overlap(&[counts.clone(), counts]), Some(1_000_000));
    }
}
