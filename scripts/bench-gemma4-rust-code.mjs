#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { spawnSync } from "node:child_process";
import http from "node:http";
import https from "node:https";

const [server, output, tasksPath, , outputCountText, concurrencyText, runsText] = process.argv.slice(2);
if (!runsText) throw new Error("usage: bench-gemma4-rust-code.mjs SERVER OUTPUT TASKS PROMPT_TOKENS OUTPUT_TOKENS CONCURRENCY RUNS");
const outputTokens = Number(outputCountText);
const concurrency = Number(concurrencyText);
const runs = Number(runsText);
const tasks = JSON.parse(fs.readFileSync(tasksPath, "utf8"));
const taskOffset = Number(process.env.TASK_OFFSET || 0);
if (!Number.isInteger(taskOffset) || taskOffset < 0 || tasks.length < taskOffset + concurrency) {
  throw new Error("task offset and concurrency exceed task file");
}
const cachePlan = (process.env.CACHE_PLAN || "").split(",").filter(Boolean);
const metricsPath = process.env.TRIAL_METRICS;
const requestTimeoutMs = Number(process.env.REQUEST_TIMEOUT_MS || 900000);

function postJson(urlText, payload) {
  const url = new URL(urlText);
  const transport = url.protocol === "https:" ? https : http;
  return new Promise((resolve, reject) => {
    const request = transport.request(url, {
      method: "POST",
      headers: { "content-type": "application/json", "content-length": Buffer.byteLength(payload) },
    }, response => {
      const chunks = [];
      response.on("data", chunk => chunks.push(chunk));
      response.on("end", () => resolve({ status: response.statusCode,
        ok: response.statusCode >= 200 && response.statusCode < 300,
        body: Buffer.concat(chunks).toString("utf8") }));
    });
    request.setTimeout(requestTimeoutMs, () => request.destroy(
      new Error(`completion timed out after ${requestTimeoutMs} ms`)));
    request.on("error", reject);
    request.end(payload);
  });
}

function readNumberMap(file) {
  return Object.fromEntries(fs.readFileSync(file, "utf8").trim().split("\n")
    .map(line => line.split(":").map(value => value.trim()))
    .map(([key, value]) => [key, Number(value)]));
}

function processFaults(pid) {
  const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8");
  const fields = stat.slice(stat.lastIndexOf(")") + 2).trim().split(/\s+/);
  return { minor_faults: Number(fields[7]), major_faults: Number(fields[9]) };
}

function residency() {
  if (!process.env.RESIDENCY_TOOL || !process.env.GEMMA4_MODEL) return null;
  const result = spawnSync(process.env.RESIDENCY_TOOL,
    ["status", process.env.GEMMA4_MODEL], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(`residency probe failed: ${result.stderr}`);
  return JSON.parse(result.stdout);
}

function snapshot(run, cacheMode, event) {
  if (!metricsPath || !process.env.SERVER_PID) return;
  const pid = Number(process.env.SERVER_PID);
  const io = readNumberMap(`/proc/${pid}/io`);
  const device = process.env.MODEL_DEVICE_STAT
    ? fs.readFileSync(process.env.MODEL_DEVICE_STAT, "utf8").trim().split(/\s+/).map(Number)
    : [];
  const record = { schema: "emufpga.gemma4-io-snapshot.v1", timestamp_ms: Date.now(),
    concurrency, run, cache_mode: cacheMode, event, server_read_bytes: io.read_bytes,
    server_rchar: io.rchar, ...processFaults(pid), device_read_ios: device[0] ?? null,
    device_read_sectors: device[2] ?? null, residency: residency() };
  fs.appendFileSync(metricsPath, `${JSON.stringify(record)}\n`);
}

function prepareCache(run, cacheMode) {
  if (cacheMode !== "cold") return;
  if (!process.env.CACHE_EVICT_TOOL) throw new Error("cold cache plan requires CACHE_EVICT_TOOL");
  const result = spawnSync(process.env.CACHE_EVICT_TOOL, ["evict"], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(`cache eviction failed for run ${run}: ${result.stderr}`);
  const after = JSON.parse(result.stdout);
  if (after.resident_pages !== 0) {
    throw new Error(`cache eviction left ${after.resident_pages} pages resident; refusing false cold label`);
  }
}

async function request(run, requestId) {
  const task = tasks[taskOffset + requestId - 1];
  const started = performance.now();
  const payload = JSON.stringify({ messages: [{ role: "user", content: task.prompt }], max_tokens: outputTokens,
    temperature: 0, seed: 42, stream: false, chat_template_kwargs: { enable_thinking: false } });
  const response = await postJson(`${server}/v1/chat/completions`, payload);
  const body = JSON.parse(response.body);
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
  const task = tasks[taskOffset + record.request_id - 1];
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
  const cacheMode = cachePlan[run - 1] ?? "untouched";
  prepareCache(run, cacheMode);
  snapshot(run, cacheMode, "start");
  const responses = await Promise.all(Array.from({ length: concurrency }, (_, id) => request(run, id + 1)));
  snapshot(run, cacheMode, "end");
  records.push(...responses.map(record => ({ ...record, cache_mode: cacheMode })));
}
const evaluated = records.map(evaluate);
fs.writeFileSync(output, evaluated.map(JSON.stringify).join("\n") + "\n");
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
console.log(JSON.stringify({ concurrency, runs, requests: evaluated.length,
  strict: evaluated.filter(item => item.strict).length,
  compiled: evaluated.filter(item => item.compiled).length,
  passed: evaluated.filter(item => item.passed).length,
  wall_ms_mean: mean(evaluated.map(item => item.wall_ms)),
  generated_tokens: evaluated.reduce((sum, item) => sum + (item.output_tokens ?? 0), 0) }));
