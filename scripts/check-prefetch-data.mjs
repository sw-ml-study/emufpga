#!/usr/bin/env node
"use strict";
import fs from "node:fs";
const d=JSON.parse(fs.readFileSync(new URL("../docs/data/prefetch-analysis.json",import.meta.url)));
if(d.schema!=="emufpga.prefetch-analysis.v1"||d.groups.length!==16)throw new Error("unexpected prefetch matrix");
for(const g of d.groups){if(g.runs!==7||g.max_error.expert>.002||g.max_error.combined>.002)throw new Error(`invalid ${g.tier}/${g.backend}`);if(!(g.phase_ms.p10<=g.phase_ms.p50&&g.phase_ms.p50<=g.phase_ms.p90))throw new Error("unordered phase distribution");if(g.backend==="prefetch"&&!Number.isFinite(g.speedup_percent))throw new Error("missing speedup");}
console.log("prefetch analysis: ok");
