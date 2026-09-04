#!/usr/bin/env node
"use strict";

import fs from "node:fs";
const [input, output, timingPath] = process.argv.slice(2);
if (!input || !output) throw new Error("usage: analyze-storage.mjs INPUT.csv OUTPUT.json [TIMING.json]");
const lines = fs.readFileSync(input, "utf8").trim().split("\n");
const provenance = Object.fromEntries(lines.filter(x => x.startsWith("# ")).map(x => x.slice(2).split(/=(.*)/s).slice(0, 2)));
const data = lines.filter(x => !x.startsWith("# "));
const headings = data.shift().split(",");
const rows = data.map(line => Object.fromEntries(line.split(",").map((value, i) => [headings[i], value])));
const percentile = (values, p) => values.map(Number).sort((a,b)=>a-b)[Math.round((values.length-1)*p)];
const groups=[];
for(const key of [...new Set(rows.map(r=>`${r.tier}|${r.mode}`))]){
  const sample=rows.filter(r=>`${r.tier}|${r.mode}`===key), throughput=sample.map(r=>Number(r.bytes)/Number(r.wall_ns));
  const wall=sample.map(r=>Number(r.wall_ns)/1e6),cpu=sample.map(r=>(Number(r.user_s)+Number(r.sys_s))*1e9/Number(r.wall_ns)*100);
  groups.push({tier:sample[0].tier,path:sample[0].path,source:sample[0].source,fstype:sample[0].fstype,mode:sample[0].mode,runs:sample.length,bytes:Number(sample[0].bytes),latency_ms:{p50:percentile(wall,.5),p10:percentile(wall,.1),p90:percentile(wall,.9)},gb_s:{p50:percentile(throughput,.5),p10:percentile(throughput,.1),p90:percentile(throughput,.9)},cpu_percent:{p50:percentile(cpu,.5),p10:percentile(cpu,.1),p90:percentile(cpu,.9)}});
}
let overlap=[];
if(timingPath){const timing=JSON.parse(fs.readFileSync(timingPath,"utf8"));for(const g of groups){const schedule=g.tier.endsWith("selected")?"selected-union":"all-expert",t=timing.groups.find(x=>x.schedule===schedule&&x.batch===1);if(!t)continue;const work=t.decode_ms.p50+t.compute_ms.p50,sequential=g.latency_ms.p50+work,ideal=Math.max(g.latency_ms.p50,work);overlap.push({tier:g.tier,mode:g.mode,storage_ms:g.latency_ms.p50,decode_compute_ms:work,synchronous_ms:sequential,ideal_double_buffered_ms:ideal,maximum_saving_percent:100*(sequential-ideal)/sequential});}}
fs.writeFileSync(output,JSON.stringify({schema:"emufpga.storage-tier-analysis.v1",provenance,caveats:["direct uses GNU dd iflag=direct as a cache-bypass proxy; it is not guaranteed to represent first access after power-on","warm uses ordinary buffered reads after an explicit priming pass","small files may measure controller and filesystem effects more than sustained media bandwidth","double-buffer values are calculated upper bounds composed from separately measured phases; the current Rust stream refills synchronously and does not yet overlap them"],groups,overlap},null,2)+"\n");
