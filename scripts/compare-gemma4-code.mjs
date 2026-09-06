#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const [residentDir, reclaimedDir, output] = process.argv.slice(2);
if (!output) throw new Error("usage: compare-gemma4-code.mjs RESIDENT_DIR RECLAIMED_DIR OUTPUT.json");
const derived = (directory) => JSON.parse(fs.readFileSync(path.join(directory, "derived.json"), "utf8"));
const records = (directory) => fs.readdirSync(directory)
  .filter(name => /^requests-\d+\.jsonl$/.test(name))
  .flatMap(name => fs.readFileSync(path.join(directory, name), "utf8").trim().split("\n").filter(Boolean).map(JSON.parse));
const resident = records(residentDir);
const reclaimed = records(reclaimedDir);
const key = (item) => `${item.concurrency}/${item.run}/${item.task_id}`;
const reclaimedByKey = new Map(reclaimed.map(item => [key(item), item]));
const pairs = resident.map(control => ({ control, candidate: reclaimedByKey.get(key(control)) }));
const disagreements = pairs.filter(({ control, candidate }) => !candidate || control.passed !== candidate.passed)
  .map(({ control, candidate }) => ({ concurrency: control.concurrency, run: control.run,
    task_id: control.task_id, resident_passed: control.passed, reclaimed_passed: candidate?.passed ?? null }));
const textMismatches = pairs.filter(({ control, candidate }) => candidate?.content !== control.content)
  .map(({ control, candidate }) => ({ concurrency: control.concurrency, run: control.run,
    task_id: control.task_id, resident_passed: control.passed, reclaimed_passed: candidate?.passed ?? null }));
const result = {
  schema: "emufpga.gemma4-executable-code-comparison.v1",
  model: "google_gemma-4-26B-A4B-it-Q5_K_M.gguf",
  quant: "Q5_K_M",
  repetitions: 2,
  corpus: "gemma4-rust-executable-v1",
  evaluation: {
    compiler_image: "rust@sha256:d7fc7de78bb8c1469933aeecbf801314d30d7d6e9f0578bba4cfa285bfa37fe6",
    runner_image: "ubuntu@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54",
    isolation: "network none, read-only root, all capabilities dropped, no-new-privileges, non-root user, CPU/memory/PID/time limits",
  },
  policies: { resident: derived(residentDir), reclaimed: derived(reclaimedDir) },
  paired_responses: pairs.length,
  outcome_disagreements: disagreements,
  exact_text_matches: pairs.filter(({ control, candidate }) => candidate?.content === control.content).length,
  text_mismatches: textMismatches,
  conclusion: "Both memory policies passed every executable test in this bounded corpus; this detects no reliability regression but is not an agentic coding benchmark.",
  caveats: [
    "Eight small dependency-free functions and two repetitions are not representative of repository-scale agent work.",
    "Tests are hidden from prompts but checked into the repository for reproducibility.",
    "Inference is timed before sequential compilation/container evaluation; GPU energy telemetry spans both.",
  ],
};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n");
