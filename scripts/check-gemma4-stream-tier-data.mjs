#!/usr/bin/env node

import fs from "node:fs";

const data = JSON.parse(fs.readFileSync(new URL("../docs/data/gemma4-stream-tier-analysis.json", import.meta.url)));
if (data.schema !== "emufpga.gemma4-stream-tiers-analysis.v1" || data.groups.length !== 8) {
  throw new Error("unexpected stream-tier schema");
}
for (const group of data.groups) {
  if (group.runs !== 7 || group.digest !== "3a7affa3b6a16627") throw new Error("invalid replay group");
  if (group.bytes !== 169_688_672 || group.buffer_bytes !== 2_097_152) throw new Error("size drift");
  if (group.cache === "cold" && group.process_read_bytes.p50 < group.bytes) throw new Error("false cold run");
  if (group.cache === "warm" && group.process_read_bytes.p50 !== 0) throw new Error("false warm run");
}
console.log("Gemma stream-tier data: ok");
