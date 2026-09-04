#!/usr/bin/env node
"use strict";

import fs from "node:fs";
import path from "node:path";

const [inputDir, output] = process.argv.slice(2);
if (!inputDir || !output) {
  throw new Error("usage: analyze-granite-routing.mjs INPUT_DIR OUTPUT.json");
}

const corpus = [
  ["hello", "Hello", [8279]],
  ["capital", "The capital of France is", [1318, 18926, 432, 45600, 438]],
  ["rust", "Write a Rust function.", [2538, 312, 19281, 667, 32]],
  ["science", "Explain photosynthesis simply.", [37394, 30680, 2817, 14425, 9639, 32]],
  ["math", "2 + 2 = ", [36, 474, 225, 36, 280, 225]],
];
const traces = corpus.map(([id, text, tokens]) => {
  const trace = JSON.parse(fs.readFileSync(path.join(inputDir, `${id}.json`), "utf8"));
  if (trace.schema !== "emufpga.moe-routing.v1" || trace.events.length !== tokens.length * 24) {
    throw new Error(`${id}: invalid schema or event count`);
  }
  for (const event of trace.events) {
    if (event.layer < 0 || event.layer >= 24 || event.token < 0 || event.token >= tokens.length ||
        !Array.isArray(event.experts) || event.experts.length !== 8 ||
        new Set(event.experts).size !== 8 || event.experts.some(x => x < 0 || x >= 32)) {
      throw new Error(`${id}: invalid route event`);
    }
  }
  return {id, text, tokens, events: trace.events};
});

const frequency = Array(32).fill(0);
for (const trace of traces) for (const event of trace.events) for (const expert of event.experts) frequency[expert]++;
const assignments = frequency.reduce((a, b) => a + b, 0);
const popularity = frequency.map((count, expert) => ({expert, count, share: count / assignments}));
popularity.sort((a, b) => b.count - a.count || a.expert - b.expert);

const unionCurve = [];
for (let batch = 1; batch <= Math.max(...traces.map(t => t.tokens.length)); batch++) {
  const samples = [];
  for (const trace of traces.filter(t => t.tokens.length >= batch)) {
    for (let layer = 0; layer < 24; layer++) {
      const experts = new Set(trace.events.filter(e => e.layer === layer && e.token < batch).flatMap(e => e.experts));
      samples.push(experts.size);
    }
  }
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  unionCurve.push({batch, samples: samples.length, observed_mean: mean,
    observed_min: Math.min(...samples), observed_max: Math.max(...samples),
    independent_mean: 32 * (1 - Math.pow(1 - 8 / 32, batch)),
    observed_traffic_fraction: mean / 32});
}

const result = {
  schema: "emufpga.moe-routing-analysis.v1",
  provenance: {
    model: "granite-3.1-1b-a400m-q6_k",
    model_sha256: "4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f",
    implementation: "Rust spm-granite-moe full 24-layer forward",
    scope: "five hand-selected prompts; prompt tokens only; no generated tokens",
  },
  corpus: traces.map(({id, text, tokens}) => ({id, text, tokens, route_events: tokens.length * 24})),
  totals: {prompts: traces.length, tokens: traces.reduce((n, t) => n + t.tokens.length, 0), layers: 24, assignments},
  popularity,
  union_curve: unionCurve,
};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n");
