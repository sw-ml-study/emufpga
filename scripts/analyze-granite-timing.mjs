#!/usr/bin/env node
"use strict";

import fs from "node:fs";
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error("usage: analyze-granite-timing.mjs INPUT.log OUTPUT.json");
const rows = fs.readFileSync(input, "utf8").split("\n").filter(line => line.startsWith("spm_q6 ")).map(line =>
  Object.fromEntries(line.split(" ").slice(1).map(field => field.split("="))));
if (!rows.length) throw new Error("no spm_q6 measurements found");
const percentile = (values, p) => {
  const sorted = values.map(Number).sort((a, b) => a - b);
  return sorted[Math.round((sorted.length - 1) * p)];
};
const groups = [];
for (const schedule of ["all-expert", "selected-union"]) for (const batch of [1, 4, 8]) {
  const sample = rows.filter(row => row.schedule === schedule && Number(row.batch) === batch);
  if (sample.length !== 7) throw new Error(`${schedule} batch ${batch}: expected 7 runs`);
  const metric = name => ({p50: percentile(sample.map(row => row[name]), .5), p90: percentile(sample.map(row => row[name]), .9), min: percentile(sample.map(row => row[name]), 0), max: percentile(sample.map(row => row[name]), 1)});
  groups.push({schedule, batch, runs: sample.length, warm_read_ms: metric("warm_read_ms"), decode_ms: metric("decode_ms"), compute_ms: metric("compute_ms"), tokens_s: metric("tokens_s")});
}
fs.writeFileSync(output, JSON.stringify({schema: "emufpga.moe-timing.v1", provenance: {model: "granite-3.1-1b-a400m-q6_k", host: "large12", mode: "release scalar Rust; warm Linux page cache; block 0 expert sweep only", caveat: "phase timers exclude router, attention, emission, cold media, and end-to-end generation"}, groups}, null, 2) + "\n");
