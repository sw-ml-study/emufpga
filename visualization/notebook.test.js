"use strict";

const assert = require("node:assert/strict");
const { LESSONS, summarizePlacement } = require("./notebook.js");

assert.deepEqual(LESSONS.blind.moved, [0, 1, 2, 3, 4, 5, 6, 7]);
assert.equal(LESSONS.blind.applications / LESSONS.blind.moved.length, 0.25);
assert.deepEqual(LESSONS.route.moved, [2, 6]);
assert.equal(LESSONS.route.applications / LESSONS.route.moved.length, 1);
assert.deepEqual(LESSONS.reuse.moved, [2, 5, 6]);
assert.equal(LESSONS.reuse.applications, 4);

const placement = summarizePlacement({throughput:[{placement:"all-gpu",parallel:8,total_seconds:{p50:80},generation_aggregate_tps:{p50:48}},{placement:"experts-cpu",parallel:8,total_seconds:{p50:400},generation_aggregate_tps:{p50:50}}],telemetry:[{placement:"all-gpu",across_runs:{peak_gpu_memory_mib:{p50:12000},gpu_energy_j:{p50:6000}}},{placement:"experts-cpu",across_runs:{peak_gpu_memory_mib:{p50:7000},gpu_energy_j:{p50:20000}}}]},8);
assert.equal(placement.cpu.time / placement.gpu.time, 5);
assert.equal(placement.gpu.vram - placement.cpu.vram, 5000);

console.log("notebook lesson model: ok");
