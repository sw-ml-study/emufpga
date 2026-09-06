#!/usr/bin/env node

import fs from "node:fs";

const fail = (message) => { throw new Error(message); };
const data = JSON.parse(fs.readFileSync("docs/data/gemma4-q5km-executable-code.json", "utf8"));
const tasks = JSON.parse(fs.readFileSync("docs/data/gemma4-rust-executable-v1.json", "utf8"));
if (tasks.length !== 8 || new Set(tasks.map(item => item.id)).size !== 8) fail("corpus must contain eight unique tasks");
for (const task of tasks) {
  if (!task.prompt.includes("fn ") || !task.tests.includes("assert")) fail(`invalid task: ${task.id}`);
  if (task.prompt.includes(task.tests)) fail(`tests leaked into prompt: ${task.id}`);
}
if (data.paired_responses !== 30 || data.outcome_disagreements.length !== 0) fail("paired outcome gate failed");
if (data.exact_text_matches !== 28 || data.text_mismatches.length !== 2) fail("text comparison changed");
for (const name of ["resident", "reclaimed"]) {
  const policy = data.policies[name];
  if (policy.response_quality.responses !== 30 || policy.response_quality.compiled !== 30 ||
      policy.response_quality.tests_passed !== 30) fail(`${name} executable gate failed`);
  if (policy.telemetry.peak_vram_mib >= 16311) fail(`${name} exceeds GPU capacity`);
  if (policy.response_quality.by_concurrency.map(item => item.concurrency).join(",") !== "1,2,4,8") {
    fail(`${name} concurrency set changed`);
  }
}
if (!data.evaluation.compiler_image.includes("sha256:") || !data.evaluation.runner_image.includes("sha256:")) {
  fail("container images are not digest-pinned");
}
console.log("Gemma executable coding data: ok");
