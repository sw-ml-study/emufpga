#!/usr/bin/env node
"use strict";

import fs from "node:fs";
const routing = JSON.parse(fs.readFileSync("docs/data/granite-routing-analysis.json", "utf8"));
const timing = JSON.parse(fs.readFileSync("docs/data/granite-timing.json", "utf8"));
if (routing.schema !== "emufpga.moe-routing-analysis.v1" || routing.totals.assignments !== routing.totals.tokens * 24 * 8) throw new Error("routing summary invariants failed");
for (const row of routing.union_curve) {
  if (!(row.observed_min <= row.observed_mean && row.observed_mean <= row.observed_max)) throw new Error("union range does not contain mean");
  const expected = 32 * (1 - Math.pow(0.75, row.batch));
  if (Math.abs(expected - row.independent_mean) > 1e-12) throw new Error("independent union formula drifted");
}
if (timing.schema !== "emufpga.moe-timing.v1" || timing.groups.length !== 6) throw new Error("timing schema failed");
for (const group of timing.groups) for (const value of Object.values(group).filter(x => typeof x === "object")) {
  if (!(value.min <= value.p50 && value.p50 <= value.max && value.min <= value.p90 && value.p90 <= value.max)) throw new Error("timing percentile outside range");
}
console.log("routing data: ok");
