#!/usr/bin/env node

import fs from "node:fs";

const [input, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-resource-service.mjs INPUT.csv OUTPUT.json");
const lines = fs.readFileSync(input, "utf8").trim().split("\n");
const keys = lines[1].split(",");
const rows = lines.slice(2).map(line => Object.fromEntries(keys.map((key, index) => [key, ["policy", "sha256"].includes(key) ? line.split(",")[index] : Number(line.split(",")[index])])));

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

const groups = [];
for (const clients of [1, 2, 4, 8]) for (const policy of ["independent", "shared"]) {
  const selected = rows.filter(row => row.clients === clients && row.policy === policy);
  groups.push({ policy, clients, runs: selected.length,
    requested_bytes: selected[0].requested_bytes, stream_bytes: selected[0].stream_bytes,
    read_bytes_p50: median(selected.map(row => row.read_bytes)),
    buffer_bytes: selected[0].buffer_bytes,
    elapsed_ms_p50: median(selected.map(row => row.elapsed_ms)),
    fairness_ratio_p50: median(selected.map(row => row.fairness_ratio)),
    sha256: selected[0].sha256 });
}
const result = { schema: "emufpga.resource-service-analysis.v1", measured_at: new Date().toISOString(), groups,
  caveats: ["This is byte-stream service work, not model arithmetic or coding quality.", "Linux page cache already coalesced concurrent independent readers into one physical HDD read.", "The shared policy removes duplicate userspace hashing and stream traversal; full expert computation is not measured.", "GPU, cgroup, NUMA, wall power, and long-lived admission fairness remain unmeasured."] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
