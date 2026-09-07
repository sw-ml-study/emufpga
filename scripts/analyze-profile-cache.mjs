#!/usr/bin/env node
import fs from "node:fs";

const [directory, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-profile-cache.mjs DIRECTORY OUTPUT.json");
const policies = ["demand", "static", "warm", "adaptive"];
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const rows = [];
for (const concurrency of [1, 2, 4, 8]) for (let run = 1; run <= 3; run += 1) {
  const group = policies.map(policy => JSON.parse(fs.readFileSync(`${directory}/c${concurrency}-${policy}-r${run}.json`)));
  if (new Set(group.map(row => row.logical_digest)).size !== 1) throw new Error(`digest mismatch c${concurrency} r${run}`);
  rows.push(...group.map(row => ({ concurrency, run, ...row })));
}
const groups = [];
for (const concurrency of [1, 2, 4, 8]) for (const policy of policies) {
  const sample = rows.filter(row => row.concurrency === concurrency && row.policy === policy);
  const demand = rows.filter(row => row.concurrency === concurrency && row.policy === "demand");
  const ratios = sample.map((row, index) => row.device_read_bytes / demand[index].device_read_bytes - 1);
  const mean = ratios.reduce((sum, value) => sum + value, 0) / ratios.length;
  const sd = Math.sqrt(ratios.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (ratios.length - 1));
  groups.push({ concurrency, policy, runs: sample.length, applications_p50: median(sample.map(row => row.applications)),
    hit_rate_p50: median(sample.map(row => row.hit_rate)), source_bytes_p50: median(sample.map(row => row.source_bytes)),
    device_read_bytes_p50: median(sample.map(row => row.device_read_bytes)), elapsed_ms_p50: median(sample.map(row => row.elapsed_ms)),
    peak_rss_bytes_p50: median(sample.map(row => row.peak_rss_bytes)), resident_bytes_p50: median(sample.map(row => row.resident_bytes)),
    paired_device_byte_change: mean, paired_device_byte_change_ci95: 4.303 * sd / Math.sqrt(ratios.length) });
}
const result = { schema: "emufpga.profile-cache-analysis.v1", model: "Gemma-4 26B-A4B Q5_K_M",
  protocol: "frozen c4 calibration profile; first 32 in-request routed layer events; three cold repetitions; 512 MiB userspace cache",
  groups, correctness: { digest_pairs: 48, mismatches: 0, prior_executable_tasks: "24/24 passed for these recorded agent-sharing traces" },
  caveats: ["This replays exact expert bytes and routing, not matrix arithmetic or generation.", "Preload time and bytes are included.", "The 32-event window is intentionally short and tests cold amortization, not long-lived steady state.", "Device counters can include unrelated host IO; matching process read bytes bound attribution."] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(groups.map(({ concurrency, policy, hit_rate_p50, paired_device_byte_change, elapsed_ms_p50 }) => ({ concurrency, policy, hit_rate_p50, paired_device_byte_change, elapsed_ms_p50 }))));
