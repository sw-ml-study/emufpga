#!/usr/bin/env node

import fs from "node:fs";

const [calibrationPath, heldoutPath, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-gemma4-expert-cache.mjs CALIBRATION.log HELDOUT.log OUTPUT.json");

function parse(path) {
  const text = fs.readFileSync(path, "utf8");
  const sizes = layerSizes(text);
  const aggregate = new Map([...text.matchAll(/^spm_mmid timestamp_s=([0-9.]+) tensor=([^ ]+) activations=(\d+)/gm)].map(match => [`${match[1]}:${match[2]}`, Number(match[3])]));
  return [...text.matchAll(/^spm_mmid_experts timestamp_s=([0-9.]+) tensor=blk\.(\d+)\.ffn_gate_up_exps\.weight counts=(.+)$/gm)].map(match => ({
    timestamp: Number(match[1]), layer: Number(match[2]), activations: aggregate.get(`${match[1]}:blk.${match[2]}.ffn_gate_up_exps.weight`),
    experts: match[3].split(",").map(pair => pair.split(":").map(Number)), bytes: sizes[Number(match[2])],
  }));
}

function layerSizes(text) {
  const sizes = Array.from({ length: 30 }, () => ({ gate: 0, down: 0 }));
  for (const match of text.matchAll(/^spm_mmid .* tensor=blk\.(\d+)\.ffn_(gate_up|down)_exps\.weight .* expert_bytes=(\d+)/gm)) {
    sizes[Number(match[1])][match[2] === "gate_up" ? "gate" : "down"] = Number(match[3]);
  }
  return sizes.map(size => size.gate + size.down);
}

function frequency(events) {
  const counts = new Map();
  for (const event of events) for (const [expert, count] of event.experts) {
    const key = `${event.layer}:${expert}`;
    counts.set(key, (counts.get(key) || 0) + count);
  }
  return counts;
}

function accesses(events) {
  return events.flatMap(event => event.experts.map(([expert, count]) => ({ key: `${event.layer}:${expert}`, count, bytes: event.bytes })));
}

function staticCache(training, sizes, demand, budget) {
  const ranked = [...training].sort((a, b) => b[1] - a[1]);
  const cached = new Set();
  let resident = 0;
  for (const [key] of ranked) {
    const bytes = sizes.get(key);
    if (bytes && resident + bytes <= budget) { cached.add(key); resident += bytes; }
  }
  return { ...score(demand, cached, resident), resources: [...cached] };
}

function adaptiveCache(demand, budget) {
  const seen = new Map(), cache = new Map();
  let resident = 0, hits = 0, byteHits = 0;
  for (let age = 0; age < demand.length; age += 1) {
    const item = demand[age], count = (seen.get(item.key) || 0) + 1;
    seen.set(item.key, count);
    if (cache.has(item.key)) { hits += 1; byteHits += item.bytes; cache.set(item.key, { count, age, bytes: item.bytes }); continue; }
    if (count < 2 || item.bytes > budget) continue;
    while (resident + item.bytes > budget && cache.size) {
      const victim = [...cache].sort((a, b) => a[1].count - b[1].count || a[1].age - b[1].age)[0];
      resident -= victim[1].bytes; cache.delete(victim[0]);
    }
    if (resident + item.bytes <= budget) { cache.set(item.key, { count, age, bytes: item.bytes }); resident += item.bytes; }
  }
  return rates(demand, hits, byteHits, resident);
}

function score(demand, cached, resident) {
  const hits = demand.filter(item => cached.has(item.key)).length;
  const byteHits = demand.filter(item => cached.has(item.key)).reduce((sum, item) => sum + item.bytes, 0);
  return rates(demand, hits, byteHits, resident);
}

function rates(demand, hits, byteHits, resident) {
  const bytes = demand.reduce((sum, item) => sum + item.bytes, 0);
  return { hit_rate: hits / demand.length, byte_hit_rate: byteHits / bytes, misses: demand.length - hits, bytes_read: bytes - byteHits, resident_bytes: resident };
}

function summary(events, concurrency) {
  const prompt = events.filter(event => event.activations > concurrency);
  const generation = events.filter(event => event.activations <= concurrency);
  const assignments = selected => selected.flatMap(event => event.experts).reduce((sum, pair) => sum + pair[1], 0);
  return { events: events.length, expert_accesses: accesses(events).length, assignments: assignments(events), prompt_assignments: assignments(prompt), generation_assignments: assignments(generation) };
}

function resourceSizes(events) {
  return new Map(accesses(events).map(item => [item.key, item.bytes]));
}

function reuseDistance(demand) {
  const last = new Map(), distances = [];
  demand.forEach((item, index) => { if (last.has(item.key)) distances.push(index - last.get(item.key)); last.set(item.key, index); });
  distances.sort((a, b) => a - b);
  return { samples: distances.length, p50_accesses: distances[Math.floor(distances.length * 0.5)], p90_accesses: distances[Math.floor(distances.length * 0.9)] };
}

const calibration = parse(calibrationPath), heldout = parse(heldoutPath);
const training = frequency(calibration), sizes = resourceSizes(calibration), demand = accesses(heldout);
const budgets = [0, 256, 512, 1024, 2048, 4096].map(mib => ({ mib,
  static: staticCache(training, sizes, demand, mib * 1048576), adaptive: adaptiveCache(demand, mib * 1048576) }));
const top = [...training].sort((a, b) => b[1] - a[1]).slice(0, 20).map(([key, count]) => ({ key, assignments: count }));
const heldoutTop = new Set([...frequency(heldout)].sort((a, b) => b[1] - a[1]).slice(0, 20).map(entry => entry[0]));
const result = { schema: "emufpga.gemma4-expert-cache.v1", source: { calibration: calibrationPath, heldout: heldoutPath, concurrency: 4, tasks_each: 4 },
  calibration: summary(calibration, 4), heldout: summary(heldout, 4), top_calibration: top,
  top20_overlap: top.filter(item => heldoutTop.has(item.key)).length / 20, heldout_reuse_distance: reuseDistance(demand), budgets,
  caveats: ["Prompt phase is inferred as activations greater than concurrency; continuous batching can blur this boundary.", "Cache policies are trace-replay simulations, not yet physical madvise or SSD placement measurements.", "A cache hit means a layer-expert resource is resident; gate/up and down bytes are counted together.", "The two four-task runs passed executable tests, but one run per split does not establish confidence intervals."] };
fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
