#!/usr/bin/env node
import fs from "node:fs";

const [concurrentPath, output, ...independentPaths] = process.argv.slice(2);
if (!output || independentPaths.length < 2) {
  throw new Error("usage: analyze-gemma4-agent-sharing.mjs CONCURRENT.log OUTPUT.json INDEPENDENT.log...");
}

function parse(path) {
  const text = fs.readFileSync(path, "utf8");
  const directory = path.slice(0, path.lastIndexOf("/"));
  const layerBytes = new Map();
  for (const match of text.matchAll(/^spm_mmid .* tensor=blk\.(\d+)\.ffn_(gate_up|down)_exps\.weight .* expert_bytes=(\d+)/gm)) {
    const layer = Number(match[1]), part = match[2], current = layerBytes.get(layer) || {};
    current[part] = Number(match[3]); layerBytes.set(layer, current);
  }
  const trace = [];
  for (const match of text.matchAll(/^spm_mmid_experts timestamp_s=([0-9.]+) tensor=blk\.(\d+)\.ffn_gate_up_exps\.weight counts=(.+)$/gm)) {
    const sizes = layerBytes.get(Number(match[2]));
    if (!sizes?.gate_up || !sizes?.down) throw new Error(`missing tensor size in ${path}`);
    const selected = match[3].split(",").map(pair => pair.split(":").map(Number));
    trace.push({ timestamp: Number(match[1]), touches: selected.length,
      assignments: selected.reduce((sum, [, count]) => sum + count, 0),
      bytes: selected.length * (sizes.gate_up + sizes.down) });
  }
  const snapshots = fs.readFileSync(`${directory}/io-snapshots.jsonl`, "utf8").trim().split("\n").map(JSON.parse);
  const summary = JSON.parse(fs.readFileSync(`${directory}/summaries.jsonl`, "utf8").trim().split("\n")[0]);
  const requests = fs.readFileSync(`${directory}/requests-${summary.concurrency}.jsonl`, "utf8").trim().split("\n").map(JSON.parse);
  const runs = [];
  for (let i = 0; i < snapshots.length; i += 2) {
    const start = snapshots[i].timestamp_ms / 1000, end = snapshots[i + 1].timestamp_ms / 1000;
    const selected = trace.filter(event => event.timestamp >= start && event.timestamp <= end);
    const generated = requests.filter(row => row.run === snapshots[i].run).reduce((sum, row) => sum + row.output_tokens, 0);
    runs.push({ run: snapshots[i].run, events: selected.length,
      expert_touches: selected.reduce((sum, row) => sum + row.touches, 0),
      assignments: selected.reduce((sum, row) => sum + row.assignments, 0),
      logical_expert_bytes: selected.reduce((sum, row) => sum + row.bytes, 0),
      elapsed_seconds: end - start, generated_tokens: generated });
  }
  const total = runs.reduce((sum, run) => ({ events: sum.events + run.events,
    expert_touches: sum.expert_touches + run.expert_touches, assignments: sum.assignments + run.assignments,
    logical_expert_bytes: sum.logical_expert_bytes + run.logical_expert_bytes,
    elapsed_seconds: sum.elapsed_seconds + run.elapsed_seconds, generated_tokens: sum.generated_tokens + run.generated_tokens,
  }), { events: 0, expert_touches: 0, assignments: 0, logical_expert_bytes: 0, elapsed_seconds: 0, generated_tokens: 0 });
  return { path, ...total, assignments_per_touch: total.assignments / total.expert_touches,
    aggregate_tokens_per_second: total.generated_tokens / total.elapsed_seconds,
    passed: summary.passed, requests: summary.requests, runs };
}

const concurrent = parse(concurrentPath);
const independent = independentPaths.map(parse);
const independentSum = independent.reduce((sum, run) => ({
  events: sum.events + run.events,
  expert_touches: sum.expert_touches + run.expert_touches,
  assignments: sum.assignments + run.assignments,
  logical_expert_bytes: sum.logical_expert_bytes + run.logical_expert_bytes,
  elapsed_seconds: sum.elapsed_seconds + run.elapsed_seconds,
  generated_tokens: sum.generated_tokens + run.generated_tokens,
  passed: sum.passed + run.passed,
  requests: sum.requests + run.requests,
}), { events: 0, expert_touches: 0, assignments: 0, logical_expert_bytes: 0,
  elapsed_seconds: 0, generated_tokens: 0, passed: 0, requests: 0 });
independentSum.assignments_per_touch = independentSum.assignments / independentSum.expert_touches;
independentSum.aggregate_tokens_per_second = independentSum.generated_tokens / independentSum.elapsed_seconds;

const pairedRepetitions = concurrent.runs.map((run, index) => {
  const base = independent.map(item => item.runs[index]).reduce((sum, item) => ({
    assignments: sum.assignments + item.assignments,
    logical_expert_bytes: sum.logical_expert_bytes + item.logical_expert_bytes,
    elapsed_seconds: sum.elapsed_seconds + item.elapsed_seconds,
    generated_tokens: sum.generated_tokens + item.generated_tokens,
  }), { assignments: 0, logical_expert_bytes: 0, elapsed_seconds: 0, generated_tokens: 0 });
  return { run: index + 1,
    logical_expert_byte_reduction: 1 - run.logical_expert_bytes / base.logical_expert_bytes,
    logical_bytes_per_assignment_reduction: 1 -
      (run.logical_expert_bytes / run.assignments) / (base.logical_expert_bytes / base.assignments),
    aggregate_tokens_per_second_gain: (run.generated_tokens / run.elapsed_seconds) /
      (base.generated_tokens / base.elapsed_seconds) - 1 };
});
function confidence(key) {
  const values = pairedRepetitions.map(row => row[key]);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const sd = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
  return { mean, ci95_half_width: 4.303 * sd / Math.sqrt(values.length) };
}

const result = { schema: "emufpga.gemma4-agent-sharing.v1", concurrent, independent, independent_sum: independentSum,
  amortization: {
    expert_touch_reduction: 1 - concurrent.expert_touches / independentSum.expert_touches,
    logical_expert_byte_reduction: 1 - concurrent.logical_expert_bytes / independentSum.logical_expert_bytes,
    assignments_per_touch_gain: concurrent.assignments_per_touch / independentSum.assignments_per_touch,
    logical_bytes_per_assignment_reduction: 1 -
      (concurrent.logical_expert_bytes / concurrent.assignments) /
      (independentSum.logical_expert_bytes / independentSum.assignments),
    assignment_count_difference: concurrent.assignments / independentSum.assignments - 1,
    aggregate_tokens_per_second_gain: concurrent.aggregate_tokens_per_second /
      independentSum.aggregate_tokens_per_second - 1,
    elapsed_time_reduction: 1 - concurrent.elapsed_seconds / independentSum.elapsed_seconds,
  },
  paired_repetitions: pairedRepetitions,
  confidence_95: {
    logical_expert_byte_reduction: confidence("logical_expert_byte_reduction"),
    logical_bytes_per_assignment_reduction: confidence("logical_bytes_per_assignment_reduction"),
    aggregate_tokens_per_second_gain: confidence("aggregate_tokens_per_second_gain"),
  },
  caveats: [
    "Logical bytes count each unique expert tensor traversal per gate-up invocation; they are not physical storage bytes.",
    "A valid comparison requires identical task prompts, output limits, model, seed, and successful outputs.",
    "Routed assignment counts can differ with batching and graph shape; bytes per assignment is the normalized comparison.",
    "This establishes runtime batching amortization, not an application-owned serial scheduler or FPGA result.",
  ] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
