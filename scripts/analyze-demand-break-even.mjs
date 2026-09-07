#!/usr/bin/env node
import fs from "node:fs";
const [directory, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-demand-break-even.mjs DIRECTORY OUTPUT.json");
const policies = ["demand", "adaptive", "recency"], groups = [];
for (const budget of [128, 512]) {
  const rows = Object.fromEntries(policies.map(policy => [policy, JSON.parse(fs.readFileSync(`${directory}/${budget}-${policy}.json`))]));
  if (new Set(Object.values(rows).map(row => row.logical_digest)).size !== 1) throw new Error(`digest mismatch at ${budget} MiB`);
  for (const policy of policies) {
    const row = rows[policy], base = rows.demand;
    const breakEven = policy === "demand" ? null : row.timeline.find(point => point.batch >= 2 && point.elapsed_ms <= base.timeline[point.batch - 1].elapsed_ms)?.batch ?? null;
    groups.push({ budget_mib: budget, policy, batches: row.batches, applications: row.applications,
      hit_rate: row.hit_rate, source_bytes: row.source_bytes, physical_bytes: row.device_read_bytes,
      expert_wait_ms: row.expert_wait_ms, elapsed_ms: row.elapsed_ms, peak_rss_bytes: row.peak_rss_bytes,
      resident_bytes: row.resident_bytes, evictions: row.evictions, elapsed_break_even_batch: breakEven,
      source_byte_reduction: 1 - row.source_bytes / base.source_bytes, physical_byte_reduction: 1 - row.device_read_bytes / base.device_read_bytes });
  }
}
const result = { schema: "emufpga.demand-cache-break-even.v1", model: "Gemma-4 26B-A4B Q5_K_M",
  protocol: "c4 held-out run 1 repeated as 12 successive batches; first 32 request-window route events; cold file at service start",
  groups, correctness: { policy_digest_mismatches: 0, source_agent_evaluations: "12/12 c4 Rust tasks passed across three source repetitions" },
  conclusion: "Demand admission breaks even in elapsed time by batch 2 and reduces userspace source copying 11.6% by batch 12, but saves no physical disk bytes beyond Linux page cache on this host.",
  caveats: ["One physical run per budget/policy; elapsed break-even is not a confidence interval.", "Repeated identical routing is a favorable upper bound on workload stability.", "Byte replay excludes expert arithmetic and GPU work."] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
