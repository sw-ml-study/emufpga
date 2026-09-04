#!/usr/bin/env node
"use strict";
import fs from "node:fs";
const data=JSON.parse(fs.readFileSync(new URL("../docs/data/storage-tier-analysis.json",import.meta.url)));
if(data.schema!=="emufpga.storage-tier-analysis.v1")throw new Error("unexpected storage schema");
if(data.groups.length<10)throw new Error("expected HDD, NVMe, and tmpfs distributions");
for(const g of data.groups){if(g.runs!==7||!(g.bytes>0))throw new Error(`invalid sample ${g.tier}`);if(!(g.gb_s.p10<=g.gb_s.p50&&g.gb_s.p50<=g.gb_s.p90))throw new Error(`unordered throughput ${g.tier}`);if(!(g.latency_ms.p10<=g.latency_ms.p50&&g.latency_ms.p50<=g.latency_ms.p90))throw new Error(`unordered latency ${g.tier}`);}
for(const x of data.overlap){const expected=Math.max(x.storage_ms,x.decode_compute_ms);if(Math.abs(x.ideal_double_buffered_ms-expected)>1e-9)throw new Error(`overlap math ${x.tier}`);}
console.log("storage analysis: ok");
