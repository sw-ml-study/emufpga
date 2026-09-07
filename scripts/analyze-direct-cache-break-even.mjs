#!/usr/bin/env node
import fs from "node:fs";
const [directory, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-direct-cache-break-even.mjs DIRECTORY OUTPUT.json");
const runs = [];
for (let run = 1; run <= 3; run += 1) {
  const demand = JSON.parse(fs.readFileSync(`${directory}/demand-r${run}.json`));
  const adaptive = JSON.parse(fs.readFileSync(`${directory}/adaptive-r${run}.json`));
  if (demand.logical_digest !== adaptive.logical_digest) throw new Error(`digest mismatch run ${run}`);
  const change = key => 1 - adaptive[key] / demand[key];
  runs.push({ run, applications: demand.applications, hits: adaptive.hits, hit_rate: adaptive.hit_rate,
    physical_byte_reduction: change("device_read_bytes"), source_byte_reduction: change("source_bytes"),
    expert_wait_reduction: change("expert_wait_ms"), elapsed_reduction: change("elapsed_ms"),
    demand_physical_bytes: demand.device_read_bytes, adaptive_physical_bytes: adaptive.device_read_bytes,
    demand_peak_rss_bytes: demand.peak_rss_bytes, adaptive_peak_rss_bytes: adaptive.peak_rss_bytes,
    break_even_batch: adaptive.timeline.find((point, index) => point.device_read_bytes < demand.timeline[index].device_read_bytes)?.batch ?? null });
}
function confidence(key) {
  const values = runs.map(row => row[key]), mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const sd = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
  return { mean, ci95_half_width: 4.303 * sd / Math.sqrt(values.length) };
}
const result = { schema: "emufpga.direct-cache-break-even.v1", model: "Gemma-4 26B-A4B Q5_K_M",
  protocol: "128 MiB admit-on-second-use; six repeated c4 held-out batches; per-file POSIX_FADV_DONTNEED verified between batches; three recorded route runs",
  runs, confidence_95: { physical_byte_reduction: confidence("physical_byte_reduction"), elapsed_reduction: confidence("elapsed_reduction"), expert_wait_reduction: confidence("expert_wait_reduction") },
  correctness: { digest_mismatches: 0, scope: "exact expert byte order; source campaign model tasks passed separately" },
  caveats: ["FADV_DONTNEED is a controlled page-cache-bypass proxy, not O_DIRECT and not memory pressure.", "Byte replay excludes expert arithmetic and GPU work.", "The same batch repeats six times, favoring stable reuse."] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
