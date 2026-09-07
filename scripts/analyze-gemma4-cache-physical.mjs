#!/usr/bin/env node
import fs from "node:fs";

const [baselineDir, staticDir, output] = process.argv.slice(2);
if (!baselineDir || !staticDir || !output) {
  throw new Error("usage: analyze-gemma4-cache-physical.mjs BASELINE STATIC OUTPUT");
}

const lines = path => fs.readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sampleSd = xs => Math.sqrt(xs.reduce((sum, x) => sum + (x - mean(xs)) ** 2, 0) / (xs.length - 1));
const metric = xs => ({ mean: mean(xs), sd: sampleSd(xs), ci95_half_width: 4.303 * sampleSd(xs) / Math.sqrt(xs.length) });

function load(dir) {
  const summary = lines(`${dir}/summaries.jsonl`)[0];
  const snapshots = lines(`${dir}/io-snapshots.jsonl`);
  const runs = [];
  for (let i = 0; i < snapshots.length; i += 2) {
    const [start, end] = snapshots.slice(i, i + 2);
    runs.push({
      device_read_bytes: (end.device_read_sectors - start.device_read_sectors) * 512,
      major_faults: end.major_faults - start.major_faults,
      resident_bytes_end: end.residency.resident_bytes,
    });
  }
  return {
    directory: dir,
    correctness: { requests: summary.requests, strict: summary.strict, compiled: summary.compiled, passed: summary.passed },
    generated_tokens: summary.generated_tokens,
    aggregate_tokens_per_second: summary.generated_tokens / (summary.wall_ms_mean * summary.runs / 1000),
    batch_wall_ms: summary.wall_ms_mean,
    runs,
    device_read_bytes: metric(runs.map(x => x.device_read_bytes)),
    major_faults: metric(runs.map(x => x.major_faults)),
    resident_bytes_end: metric(runs.map(x => x.resident_bytes_end)),
  };
}

const baseline = load(baselineDir), static1g = load(staticDir);
const result = {
  schema: "emufpga.gemma4-expert-cache-physical.v1",
  protocol: "held-out Rust coding tasks; concurrency 4; three repetitions; cold before server load; llama-server warmup enabled; cache state otherwise untouched",
  baseline,
  static_1g: static1g,
  deltas: {
    batch_wall_percent: (static1g.batch_wall_ms / baseline.batch_wall_ms - 1) * 100,
    aggregate_tokens_per_second_percent: (static1g.aggregate_tokens_per_second / baseline.aggregate_tokens_per_second - 1) * 100,
    device_read_bytes_percent: (static1g.device_read_bytes.mean / baseline.device_read_bytes.mean - 1) * 100,
  },
  caveat: "Three sequential repetitions are not independent cold trials. The first repetition dominates physical I/O; later repetitions benefit from the Linux page cache. Application MADV_DONTNEED drops mappings, not necessarily clean file-cache pages.",
};
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
