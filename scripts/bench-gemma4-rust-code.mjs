#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { spawnSync } from "node:child_process";

const [server, output, tasksPath, , outputCountText, concurrencyText, runsText] = process.argv.slice(2);
if (!runsText) throw new Error("usage: bench-gemma4-rust-code.mjs SERVER OUTPUT TASKS PROMPT_TOKENS OUTPUT_TOKENS CONCURRENCY RUNS");
const outputTokens = Number(outputCountText);
const concurrency = Number(concurrencyText);
const runs = Number(runsText);
const tasks = JSON.parse(fs.readFileSync(tasksPath, "utf8"));
if (tasks.length < concurrency) throw new Error("task file has fewer tasks than concurrency");

async function request(run, requestId) {
  const task = tasks[requestId - 1];
  const started = performance.now();
  const response = await fetch(`${server}/v1/chat/completions`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: task.prompt }], max_tokens: outputTokens,
      temperature: 0, seed: 42, stream: false, chat_template_kwargs: { enable_thinking: false } }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`completion: HTTP ${response.status}: ${JSON.stringify(body)}`);
  const choice = body.choices?.[0];
  const message = choice?.message ?? {};
  return { schema: "emufpga.executable-rust-response.v1", run, concurrency, request_id: requestId,
    task_id: task.id, wall_ms: performance.now() - started,
    prompt_tokens: body.usage?.prompt_tokens ?? null, output_tokens: body.usage?.completion_tokens ?? null,
    finish_reason: choice?.finish_reason ?? null,
    content: message.content || message.reasoning_content || "",
    had_reasoning_content: Boolean(message.reasoning_content) };
}

function evaluate(record) {
  const task = tasks[record.request_id - 1];
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "emufpga-eval-input-"));
  try {
    const responsePath = path.join(directory, "response.txt");
    const testsPath = path.join(directory, "tests.txt");
    const resultPath = path.join(directory, "result.json");
    fs.writeFileSync(responsePath, record.content);
    fs.writeFileSync(testsPath, task.tests);
    const evaluation = spawnSync("node", ["scripts/evaluate-rust-candidate.mjs", responsePath, testsPath, resultPath],
      { encoding: "utf8", timeout: 40000 });
    if (evaluation.status !== 0) throw new Error(`evaluator failed: ${evaluation.stderr}`);
    return { ...record, ...JSON.parse(fs.readFileSync(resultPath, "utf8")) };
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
}

const records = [];
for (let run = 1; run <= runs; run += 1) {
  const responses = await Promise.all(Array.from({ length: concurrency }, (_, id) => request(run, id + 1)));
  records.push(...responses.map(evaluate));
}
fs.writeFileSync(output, records.map(JSON.stringify).join("\n") + "\n");
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
console.log(JSON.stringify({ concurrency, runs, requests: records.length,
  strict: records.filter(item => item.strict).length, compiled: records.filter(item => item.compiled).length,
  passed: records.filter(item => item.passed).length,
  wall_ms_mean: mean(records.map(item => item.wall_ms)),
  generated_tokens: records.reduce((sum, item) => sum + (item.output_tokens ?? 0), 0) }));
