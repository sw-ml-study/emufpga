"use strict";
const assert=require("node:assert/strict"),{nodes,scenarios}=require("./architectures.js");
assert.equal(Object.keys(scenarios).length,5);
for(const [id,s] of Object.entries(scenarios)){assert.ok(s.edges.length>=5,`${id} has useful flows`);for(const e of s.edges){assert.ok(nodes[e.from]&&nodes[e.to],`${id}: known endpoints`);assert.ok(["weights","activation","kv","control"].includes(e.kind));}}
assert.match(scenarios.pio.thesis,/cannot/i);assert.match(scenarios.fpga.state,/UNMEASURED|SIMULATED/);
console.log("architecture scenarios: ok");
