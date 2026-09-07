#!/usr/bin/env node
import fs from "node:fs";

const read = name => JSON.parse(fs.readFileSync(`docs/data/${name}`, "utf8"));
const curve = read("gemma4-agent-sharing-curve.json");
const c8 = read("gemma4-agent-sharing-c8-r3.json");
if (curve.schema !== "emufpga.gemma4-agent-sharing-curve.v1" || curve.repetitions !== 3) throw new Error("bad sharing curve schema");
if (curve.curve.map(row => row.concurrency).join(",") !== "1,2,4") throw new Error("missing c1/c2/c4 curve");
for (const row of curve.curve) {
  if (row.passed !== row.requests || row.generated_tokens !== 690) throw new Error(`quality mismatch at c${row.concurrency}`);
}
if (!(curve.curve[1].logical_expert_byte_reduction > 0.1 && curve.curve[2].logical_expert_byte_reduction > 0.2)) throw new Error("sharing reduction regressed");
if (!(curve.curve[2].aggregate_tokens_per_second > curve.curve[1].aggregate_tokens_per_second)) throw new Error("c4 aggregate rate regressed");
if (c8.concurrent.passed !== 24 || c8.independent_sum.passed !== 24) throw new Error("c8 quality mismatch");
if (!(c8.amortization.logical_bytes_per_assignment_reduction > 0.1)) throw new Error("c8 normalized sharing regressed");
console.log("Gemma multi-agent sharing data: ok");
