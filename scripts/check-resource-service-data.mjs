#!/usr/bin/env node

import fs from "node:fs";

const data = JSON.parse(fs.readFileSync(new URL("../docs/data/resource-service-analysis.json", import.meta.url)));
if (data.schema !== "emufpga.resource-service-analysis.v1" || data.groups.length !== 8) throw new Error("unexpected resource data");
for (const row of data.groups) {
  if (row.runs !== 3 || row.sha256 !== "453cc38f761e0bb8f6af101081aa969b3938a8d0d7a220a2408131a8dbf8cbfd") throw new Error("resource measurement drift");
  if (row.read_bytes_p50 !== 169689088) throw new Error("cold physical bytes drift");
  const expected = row.policy === "shared" ? 169688672 : 169688672 * row.clients;
  if (row.stream_bytes !== expected) throw new Error("scheduler stream-byte drift");
}
console.log("resource service data: ok");
