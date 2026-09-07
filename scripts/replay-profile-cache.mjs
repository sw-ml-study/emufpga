#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import { performance } from "node:perf_hooks";

const [manifestPath, modelPath, tracePath, policy, budgetText, limitText, output] = process.argv.slice(2);
if (!output) throw new Error("usage: replay-profile-cache.mjs MANIFEST MODEL TRACE demand|static|warm|adaptive|recency BUDGET_MIB MAX_EVENTS OUTPUT");
const budget = Number(budgetText) * 1048576;
const limit = Number(limitText);
const batches = Number(process.env.BATCHES || 1);
if (!Number.isSafeInteger(budget) || !Number.isSafeInteger(limit) || limit < 1) throw new Error("invalid bounds");
if (!["demand", "static", "warm", "adaptive", "recency"].includes(policy)) throw new Error("invalid policy");

const envelope = JSON.parse(fs.readFileSync(manifestPath));
const profile = envelope.payload;
const tensors = new Map();
for (const tensor of profile.inventory.tensors) {
  const rows = tensors.get(tensor.layer) || [];
  rows.push(tensor);
  tensors.set(tensor.layer, rows);
}
const tracePaths = tracePath.split(",");
const runIndex = Number(process.env.RUN_INDEX || 1) - 1;
function requestWindow(path) {
  const snapshots = fs.readFileSync(`${path.slice(0, path.lastIndexOf("/"))}/io-snapshots.jsonl`, "utf8").trim().split("\n").map(JSON.parse);
  return [snapshots[runIndex * 2].timestamp_ms / 1000, snapshots[runIndex * 2 + 1].timestamp_ms / 1000];
}
const parsed = tracePaths.map(path => {
  const [start, end] = requestWindow(path);
  return [...fs.readFileSync(path, "utf8").matchAll(/^spm_mmid_experts timestamp_s=([0-9.]+) tensor=blk\.(\d+)\.ffn_gate_up_exps\.weight counts=(.+)$/gm)]
    .filter(match => Number(match[1]) >= start && Number(match[1]) <= end)
    .map(match => match[3].split(",").map(pair => `${match[2]}:${pair.split(":")[0]}`));
});
const events = Array.from({ length: limit }, (_, index) => parsed[index % parsed.length][Math.floor(index / parsed.length)]).filter(Boolean);
if (!events.length) throw new Error("trace has no expert events");

const fd = fs.openSync(modelPath, "r");
const cache = new Map();
const seen = new Map();
const digests = new Map();
let resident = 0, age = 0, hits = 0, misses = 0, evictions = 0, sourceBytes = 0, peakRss = 0, expertWaitMs = 0;
const logical = crypto.createHash("sha256");
const io = () => Object.fromEntries(fs.readFileSync("/proc/self/io", "utf8").trim().split("\n").map(line => line.split(": ")).map(([key, value]) => [key, Number(value)]));
const device = () => process.env.MODEL_DEVICE_STAT ? fs.readFileSync(process.env.MODEL_DEVICE_STAT, "utf8").trim().split(/\s+/).map(Number) : [];

function load(key) {
  const started = performance.now();
  const [layer, expert] = key.split(":").map(Number);
  const chunks = tensors.get(layer).map(tensor => {
    const data = Buffer.allocUnsafe(tensor.bytes_per_expert);
    const count = fs.readSync(fd, data, 0, data.length, tensor.offset + expert * tensor.bytes_per_expert);
    if (count !== data.length) throw new Error(`short expert read ${key}`);
    return data;
  });
  const data = Buffer.concat(chunks);
  sourceBytes += data.length;
  expertWaitMs += performance.now() - started;
  let digest = digests.get(key);
  if (!digest) {
    digest = crypto.createHash("sha256").update(data).digest("hex");
    digests.set(key, digest);
  }
  return { data, digest, age: age++, uses: 1 };
}

function evictFor(bytes, protectedKeys) {
  while (resident + bytes > budget) {
    const candidates = [...cache].filter(([key]) => !protectedKeys.has(key));
    if (!candidates.length) return false;
    candidates.sort((a, b) => a[1].uses - b[1].uses || a[1].age - b[1].age);
    const [key, value] = candidates[0];
    cache.delete(key); resident -= value.data.length; evictions += 1;
  }
  return true;
}

const staticKeys = new Set(profile.placement.preload);
const warmKeys = new Set([...staticKeys, ...profile.placement.warm_candidates]);
const fixed = policy === "static" ? staticKeys : policy === "warm" ? warmKeys : new Set();
const startedIo = io(), startedDevice = device();
const started = performance.now();
for (const key of fixed) {
  const value = load(key);
  if (!evictFor(value.data.length, fixed)) break;
  cache.set(key, value); resident += value.data.length;
}
const timeline = [];
for (let batch = 1; batch <= batches; batch += 1) for (const event of events) for (const key of event) {
  let value = cache.get(key);
  if (value) { hits += 1; value.uses += 1; value.age = age++; }
  else {
    misses += 1; value = load(key);
    const count = (seen.get(key) || 0) + 1;
    seen.set(key, count);
    if (((policy === "adaptive" && count >= 2) || policy === "recency") && evictFor(value.data.length, fixed)) {
      cache.set(key, value); resident += value.data.length;
    }
  }
  logical.update(key).update(value.digest);
  peakRss = Math.max(peakRss, process.memoryUsage.rss());
  if (timeline.length < batch && event === events.at(-1) && key === event.at(-1)) {
    const currentIo = io(), currentDevice = device();
    timeline.push({ batch, hits, misses, evictions, source_bytes: sourceBytes,
      process_read_bytes: currentIo.read_bytes - startedIo.read_bytes,
      process_rchar: currentIo.rchar - startedIo.rchar,
      device_read_bytes: currentDevice.length ? (currentDevice[2] - startedDevice[2]) * 512 : null,
      resident_bytes: resident, expert_wait_ms: expertWaitMs, elapsed_ms: performance.now() - started });
  }
}
const endedIo = io(), endedDevice = device();
fs.closeSync(fd);
const result = { schema: "emufpga.profile-cache-replay.v1", policy, budget_bytes: budget,
  batches, events_per_batch: events.length, applications: events.flat().length * batches, hits, misses, evictions,
  hit_rate: hits / (events.flat().length * batches), source_bytes: sourceBytes,
  process_read_bytes: endedIo.read_bytes - startedIo.read_bytes,
  read_syscalls: endedIo.syscr - startedIo.syscr,
  device_read_ios: endedDevice.length ? endedDevice[0] - startedDevice[0] : null,
  device_read_bytes: endedDevice.length ? (endedDevice[2] - startedDevice[2]) * 512 : null,
  resident_bytes: resident, peak_rss_bytes: peakRss, expert_wait_ms: expertWaitMs, timeline,
  elapsed_ms: performance.now() - started, logical_digest: logical.digest("hex"),
  manifest_payload_sha256: envelope.payload_sha256,
  trace_sha256: tracePaths.map(path => crypto.createHash("sha256").update(fs.readFileSync(path)).digest("hex")) };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result));
