#!/usr/bin/env node

import fs from "node:fs";

const [input, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-gemma4-stream-tiers.mjs INPUT.csv OUTPUT.json");
const lines = fs.readFileSync(input, "utf8").trim().split("\n");
const rows = lines.slice(2).map(line => parse(lines[1].split(","), line.split(",")));
const expectedDigest = rows[0]?.digest;
if (!expectedDigest || rows.some(row => row.digest !== expectedDigest)) throw new Error("digest mismatch");

function parse(keys, values) {
  return Object.fromEntries(keys.map((key, index) => [key,
    ["tier", "cache", "backend", "digest"].includes(key) ? values[index] : Number(values[index])]));
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * fraction)];
}

function distribution(values) {
  return { p10: percentile(values, 0.1), p50: percentile(values, 0.5), p90: percentile(values, 0.9) };
}

const groups = [];
for (const tier of ["hdd", "nvme"]) for (const cache of ["cold", "warm"]) {
  for (const backend of ["sync", "prefetch"]) {
    const selected = rows.filter(row => row.tier === tier && row.cache === cache && row.backend === backend);
    groups.push({ tier, cache, backend, runs: selected.length, digest: expectedDigest,
      bytes: selected[0].bytes, buffer_bytes: selected[0].buffer_bytes,
      elapsed_ms: distribution(selected.map(row => row.elapsed_ms)),
      bandwidth_mib_s: distribution(selected.map(row => row.bandwidth_mib_s)),
      process_read_bytes: distribution(selected.map(row => row.process_read_bytes)),
      device_read_ios: distribution(selected.map(row => row.device_read_ios)),
      device_read_bytes: distribution(selected.map(row => row.device_read_bytes)),
      device_average_read_bytes: distribution(selected.map(row => row.device_read_ios ? row.device_read_bytes / row.device_read_ios : 0)) });
  }
}

const result = { schema: "emufpga.gemma4-stream-tiers-analysis.v1", captured_utc: new Date().toISOString(),
  source_sha256: "453cc38f761e0bb8f6af101081aa969b3938a8d0d7a220a2408131a8dbf8cbfd",
  groups, caveats: [
    "The artifact is a validated Gemma-4 layer-0 selected-expert trace slice, not full inference.",
    "Device counters are idle-device upper bounds; process read_bytes is attributable.",
    "The digest consumer is deterministic byte work, while activation correctness was validated separately.",
    "No parallel-SAS or SSD-hot/HDD-cold expert segmentation was available on this host.",
  ] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(groups, null, 2));
