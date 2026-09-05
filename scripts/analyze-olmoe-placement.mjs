#!/usr/bin/env node
"use strict";

import fs from "node:fs";
import path from "node:path";

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  throw new Error("usage: analyze-olmoe-placement.mjs INPUT_DIR OUTPUT.json");
}

const percentile = (values, fraction) => {
  const sorted = values.map(Number).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  return sorted[Math.round((sorted.length - 1) * fraction)];
};

const distribution = (values) => ({
  min: Math.min(...values),
  p50: percentile(values, 0.5),
  max: Math.max(...values),
});

const manifest = JSON.parse(fs.readFileSync(path.join(input, "manifest.json"), "utf8"));
const results = fs.readFileSync(path.join(input, "results.jsonl"), "utf8")
  .trim().split("\n").filter(Boolean).map(JSON.parse);

const keys = [...new Set(results.map((row) => `${row.placement}|${row.pl}`))];
const throughput = keys.map((key) => {
  const rows = results.filter((row) => `${row.placement}|${row.pl}` === key);
  const parallel = rows[0].pl;
  return {
    placement: rows[0].placement,
    parallel,
    runs: rows.length,
    prompt_tps: distribution(rows.map((row) => row.speed_pp)),
    generation_aggregate_tps: distribution(rows.map((row) => row.speed_tg)),
    generation_per_request_tps: distribution(rows.map((row) => row.speed_tg / parallel)),
    total_seconds: distribution(rows.map((row) => row.t)),
  };
});

const placements = [...new Set(results.map((row) => row.placement))];
const telemetry = placements.map((placement) => {
  const runs = [...new Set(results.filter((row) => row.placement === placement).map((row) => row.run))];
  const summaries = runs.map((run) => {
    const stem = path.join(input, `${placement}-run-${run}`);
    const rss = fs.readFileSync(`${stem}-rss-kib.csv`, "utf8").trim()
      .split("\n").filter(Boolean).map(Number).filter(Number.isFinite);
    const gpu = fs.readFileSync(`${stem}-gpu.csv`, "utf8").trim()
      .split("\n").filter(Boolean).map((line) => {
        const fields = line.split(",").map((field) => field.trim());
        return {
          time_ms: Date.parse(fields[0]),
          memory_mib: Number(fields[1]),
          utilization_percent: Number(fields[2]),
          power_w: Number(fields[3]),
        };
      }).filter((row) => Object.values(row).every(Number.isFinite));
    let energy_j = 0;
    for (let index = 1; index < gpu.length; index += 1) {
      const seconds = (gpu[index].time_ms - gpu[index - 1].time_ms) / 1000;
      if (seconds >= 0 && seconds < 5) {
        energy_j += seconds * (gpu[index].power_w + gpu[index - 1].power_w) / 2;
      }
    }
    return {
      run,
      peak_process_rss_mib: Math.max(...rss) / 1024,
      peak_gpu_memory_mib: Math.max(...gpu.map((row) => row.memory_mib)),
      mean_gpu_utilization_percent: gpu.reduce((sum, row) => sum + row.utilization_percent, 0) / gpu.length,
      gpu_energy_j: energy_j,
      telemetry_seconds: (gpu.at(-1).time_ms - gpu[0].time_ms) / 1000,
    };
  });
  return {
    placement,
    runs: summaries,
    across_runs: {
      peak_process_rss_mib: distribution(summaries.map((row) => row.peak_process_rss_mib)),
      peak_gpu_memory_mib: distribution(summaries.map((row) => row.peak_gpu_memory_mib)),
      gpu_energy_j: distribution(summaries.map((row) => row.gpu_energy_j)),
    },
  };
});

const report = {
  schema: "emufpga.olmoe-placement-analysis.v1",
  manifest,
  throughput,
  telemetry,
  caveats: [
    "llama-batched-bench uses synthetic batched tokens, not independent semantic tasks",
    "each telemetry run spans model loading and the complete 1/2/4/8 sweep, so telemetry is placement-run scoped, not concurrency-row scoped",
    "gpu_energy_j integrates NVIDIA board power samples and is not whole-system wall energy",
    "process RSS does not include every kernel page-cache or driver allocation",
    "experts-cpu is conventional llama.cpp tensor placement, not the ordered bounded serial implementation",
  ],
};

fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
