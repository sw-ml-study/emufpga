#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const [input, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-gemma4-offload.mjs INPUT_DIR OUTPUT.json");
const manifest = JSON.parse(fs.readFileSync(path.join(input, "manifest.json"), "utf8"));
const summaries = fs.readFileSync(path.join(input, "summaries.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
const telemetry = fs.readFileSync(path.join(input, "telemetry.csv"), "utf8").trim().split("\n").map((line) => {
  const [seconds, rssKib, vramMib, utilization, powerWatts] = line.split(",").map(Number);
  return { seconds, rssKib, vramMib, utilization, powerWatts };
});
let gpuEnergyJoules = 0;
for (let index = 1; index < telemetry.length; index += 1) {
  const previous = telemetry[index - 1];
  const current = telemetry[index];
  gpuEnergyJoules += (current.seconds - previous.seconds) * (current.powerWatts + previous.powerWatts) / 2;
}
const result = {
  schema: "emufpga.gemma4-offload-derived.v1",
  captured_utc: new Date().toISOString(),
  manifest,
  telemetry: {
    scope: "server load plus complete 1/2/4/8 request sweep; NVIDIA board energy only",
    samples: telemetry.length,
    elapsed_seconds: telemetry.at(-1).seconds - telemetry[0].seconds,
    peak_process_rss_mib: Math.max(...telemetry.map((item) => item.rssKib)) / 1024,
    peak_vram_mib: Math.max(...telemetry.map((item) => item.vramMib)),
    peak_gpu_utilization_percent: Math.max(...telemetry.map((item) => item.utilization)),
    gpu_board_energy_joules: gpuEnergyJoules,
  },
  requests: summaries,
  caveats: [
    "Simple deterministic correctness probes are smoke tests, not a benchmark of model quality.",
    "TTFT is client-observed; inter-token intervals are timestamps of streamed content events.",
    "Energy excludes CPU, DRAM, storage, motherboard, fans, and PSU losses.",
    "The conventional control keeps 20 of 30 repeating layers on GPU; it is not ordered expert streaming.",
  ],
};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result.telemetry));
