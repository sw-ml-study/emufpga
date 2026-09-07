#!/usr/bin/env node
import fs from "node:fs";

const [c2aPath, c2bPath, c4Path, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-gemma4-sharing-curve.mjs C2A.json C2B.json C4.json OUTPUT.json");
const read = path => JSON.parse(fs.readFileSync(path, "utf8"));
const c2a = read(c2aPath), c2b = read(c2bPath), c4 = read(c4Path);
const keys = ["events", "expert_touches", "assignments", "logical_expert_bytes", "elapsed_seconds", "generated_tokens", "passed", "requests"];
const add = rows => Object.fromEntries(keys.map(key => [key, rows.reduce((sum, row) => sum + row[key], 0)]));
const baseline = add([c2a.independent_sum, c2b.independent_sum]);
const summarize = (concurrency, row) => ({ concurrency, ...row,
  logical_expert_byte_reduction: 1 - row.logical_expert_bytes / baseline.logical_expert_bytes,
  logical_bytes_per_assignment_reduction: 1 -
    (row.logical_expert_bytes / row.assignments) / (baseline.logical_expert_bytes / baseline.assignments),
  elapsed_time_reduction: 1 - row.elapsed_seconds / baseline.elapsed_seconds,
  aggregate_tokens_per_second: row.generated_tokens / row.elapsed_seconds,
  aggregate_tokens_per_second_gain: (row.generated_tokens / row.elapsed_seconds) /
    (baseline.generated_tokens / baseline.elapsed_seconds) - 1,
});
const result = { schema: "emufpga.gemma4-agent-sharing-curve.v1", repetitions: 3,
  baseline: summarize(1, baseline),
  curve: [summarize(1, baseline), summarize(2, add([c2a.concurrent, c2b.concurrent])), summarize(4, c4.concurrent)],
  c4_confidence_95: c4.confidence_95,
  c2_task_mix: [c2a.amortization, c2b.amortization],
  caveats: ["Concurrency 2 is two disjoint two-task batches; concurrency 4 is all four tasks together.",
    "Results cover four small executable Rust tasks, not repository-scale autonomous agents.",
    "Logical expert bytes are tensor traversal demand, not measured storage traffic."],
};
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
