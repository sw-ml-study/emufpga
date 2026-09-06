#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const [input, output] = process.argv.slice(2);
if (!output) throw new Error("usage: summarize-gemma4-cold-io.mjs INPUT_DIR OUTPUT.json");
const campaign = JSON.parse(fs.readFileSync(path.join(input, "campaign.json"), "utf8"));
const directories = fs.readdirSync(input, { withFileTypes: true })
  .filter(entry => entry.isDirectory()).map(entry => entry.name).sort();
const trials = directories.map(directory => {
  const match = /^r(\d+)-c(\d+)-(cold|warm)-(resident|reclaimed)$/.exec(directory);
  if (!match) throw new Error(`unexpected campaign directory: ${directory}`);
  const data = JSON.parse(fs.readFileSync(path.join(input, directory, "derived.json"), "utf8"));
  const request = data.physical_io.trials[0];
  return { repetition: Number(match[1]), concurrency: Number(match[2]),
    cache_mode: match[3], policy: match[4], manifest: data.manifest,
    startup: data.physical_io.startup, request, telemetry: data.telemetry,
    quality: data.response_quality, expert_trace: data.expert_trace };
});

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const groups = [];
for (const concurrency of [...new Set(trials.map(item => item.concurrency))].sort((a, b) => a - b)) {
  for (const cacheMode of ["cold", "warm"]) {
    for (const policy of ["resident", "reclaimed"]) {
      const selected = trials.filter(item => item.concurrency === concurrency &&
        item.cache_mode === cacheMode && item.policy === policy);
      if (selected.length === 0) continue;
      groups.push({ concurrency, cache_mode: cacheMode, policy, trials: selected.length,
        tests_passed: selected.reduce((sum, item) => sum + item.quality.tests_passed, 0),
        requests: selected.reduce((sum, item) => sum + item.quality.responses, 0),
        startup_read_bytes_mean: mean(selected.map(item => item.startup.server_read_bytes)),
        request_read_bytes_mean: mean(selected.map(item => item.request.server_read_bytes)),
        request_read_ios_mean: mean(selected.map(item => item.request.device_read_ios_upper_bound)),
        request_read_size_mean: mean(selected.map(item => item.request.device_average_read_bytes_upper_bound ?? 0)),
        request_wall_ms_mean: mean(selected.map(item => item.request.request_wall_ms_max)),
        sweep_elapsed_ms_mean: mean(selected.map(item => item.request.elapsed_ms)),
        peak_rss_mib_mean: mean(selected.map(item => item.telemetry.peak_process_rss_mib)),
        peak_vram_mib: Math.max(...selected.map(item => item.telemetry.peak_vram_mib)),
        gpu_board_energy_joules_mean: mean(selected.map(item => item.telemetry.gpu_board_energy_joules)),
        generated_tokens: selected.reduce((sum, item) => sum + item.request.generated_tokens, 0),
        generated_tokens_per_second: selected.reduce((sum, item) => sum + item.request.generated_tokens, 0) /
          selected.reduce((sum, item) => sum + item.request.elapsed_ms, 0) * 1000,
        passing_tasks_per_hour: selected.reduce((sum, item) => sum + item.quality.tests_passed, 0) /
          selected.reduce((sum, item) => sum + item.request.elapsed_ms, 0) * 3_600_000,
        request_read_bytes_per_generated_token: selected.reduce((sum, item) => sum + item.request.server_read_bytes, 0) /
          selected.reduce((sum, item) => sum + item.request.generated_tokens, 0),
        request_read_bytes_per_passing_task: selected.reduce((sum, item) => sum + item.request.server_read_bytes, 0) /
          selected.reduce((sum, item) => sum + item.quality.tests_passed, 0),
        logical_expert_bytes: selected.reduce((sum, item) => sum + (item.request.expert_trace?.logical_bytes ?? 0), 0),
      });
    }
  }
}

const result = { schema: "emufpga.gemma4-cold-io-summary.v1",
  captured_utc: new Date().toISOString(), campaign, trials, groups,
  caveats: [
    "The model resides on one shared HGST HDD; process read_bytes is attributable while device counters are upper bounds.",
    "mmap page faults are the control access pattern, not the proposed execution-order sequential stream.",
    "Speed is usability evidence, not the capacity success criterion.",
  ] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(groups, null, 2));
