#!/usr/bin/env node

import fs from "node:fs";
import { performance } from "node:perf_hooks";

const [server, output, tasksPath, promptCountText, outputCountText, concurrencyText, runsText] = process.argv.slice(2);
if (!runsText) {
  throw new Error("usage: bench-gemma4-requests.mjs SERVER OUTPUT TASKS.json PROMPT_TOKENS OUTPUT_TOKENS CONCURRENCY RUNS");
}
const promptTokens = Number(promptCountText);
const outputTokens = Number(outputCountText);
const concurrency = Number(concurrencyText);
const runs = Number(runsText);
for (const [name, value] of Object.entries({ promptTokens, outputTokens, concurrency, runs })) {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
}

async function jsonPost(path, body) {
  const response = await fetch(`${server}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}

const tasks = JSON.parse(fs.readFileSync(tasksPath, "utf8"));
if (!Array.isArray(tasks) || tasks.length < concurrency) throw new Error("task file has fewer tasks than concurrency");
const filler = (await jsonPost("/tokenize", {
  content: "Carefully follow the final instruction. Relevant facts are unchanged. ",
  add_special: false,
})).tokens;
if (!Array.isArray(filler) || filler.length === 0) throw new Error("tokenizer returned no filler tokens");
const prepared = await Promise.all(tasks.slice(0, concurrency).map(async (item) => {
  if (typeof item.prompt !== "string" || typeof item.expected !== "string") throw new Error("invalid task record");
  const task = (await jsonPost("/tokenize", { content: item.prompt, add_special: true })).tokens;
  if (!Array.isArray(task) || task.length > promptTokens) throw new Error("task exceeds fixed prompt length");
  const prompt = [];
  while (prompt.length < promptTokens - task.length) prompt.push(filler[prompt.length % filler.length]);
  prompt.push(...task);
  if (prompt.length !== promptTokens) throw new Error("fixed prompt construction failed");
  return { ...item, tokens: prompt };
}));

async function request(run, requestId) {
  const task = prepared[requestId - 1];
  const started = performance.now();
  const response = await fetch(`${server}/completion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      prompt: task.tokens,
      n_predict: outputTokens,
      ignore_eos: true,
      temperature: 0,
      cache_prompt: false,
      stream: true,
    }),
  });
  if (!response.ok || !response.body) throw new Error(`completion: HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let content = "";
  let first;
  let previous;
  let final;
  const interTokenMs = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    pending += decoder.decode(value, { stream: true });
    const events = pending.split("\n\n");
    pending = events.pop() ?? "";
    for (const event of events) {
      const line = event.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const item = JSON.parse(line.slice(6));
      const now = performance.now();
      if (item.content) {
        first ??= now;
        if (previous !== undefined) interTokenMs.push(now - previous);
        previous = now;
        content += item.content;
      }
      if (item.stop) final = item;
    }
  }
  const ended = performance.now();
  if (first === undefined || !final) throw new Error("stream ended without timing data");
  return {
    schema: "emufpga.independent-request.v1",
    run,
    concurrency,
    request_id: requestId,
    prompt_tokens: promptTokens,
    output_tokens: outputTokens,
    expected: task.expected,
    correct: content.trimStart().toLocaleLowerCase().startsWith(task.expected.toLocaleLowerCase()),
    ttft_ms: first - started,
    wall_ms: ended - started,
    inter_token_ms: interTokenMs,
    server_timings: final.timings,
    content,
  };
}

const records = [];
for (let run = 1; run <= runs; run += 1) {
  const group = await Promise.all(Array.from({ length: concurrency }, (_, id) => request(run, id + 1)));
  records.push(...group);
}
fs.writeFileSync(output, records.map((item) => JSON.stringify(item)).join("\n") + "\n");
const ttft = records.map((item) => item.ttft_ms).sort((a, b) => a - b);
const intervals = records.flatMap((item) => item.inter_token_ms).sort((a, b) => a - b);
const percentile = (values, fraction) => values[Math.max(0, Math.ceil(values.length * fraction) - 1)];
const totalTokens = records.length * outputTokens;
const runWall = new Map();
for (const item of records) runWall.set(item.run, Math.max(runWall.get(item.run) ?? 0, item.wall_ms));
const aggregateTps = [...runWall.values()].map((ms) => concurrency * outputTokens * 1000 / ms);
console.log(JSON.stringify({
  concurrency,
  runs,
  requests: records.length,
  correct: records.filter((item) => item.correct).length,
  generated_tokens: totalTokens,
  aggregate_tps_mean: aggregateTps.reduce((sum, item) => sum + item, 0) / aggregateTps.length,
  per_request_tps_mean: records.reduce((sum, item) => sum + outputTokens * 1000 / item.wall_ms, 0) / records.length,
  ttft_ms_p50: percentile(ttft, 0.5),
  ttft_ms_p95: percentile(ttft, 0.95),
  inter_token_ms_p50: percentile(intervals, 0.5),
  inter_token_ms_p95: percentile(intervals, 0.95),
}));
