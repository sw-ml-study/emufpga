#!/usr/bin/env node

import fs from "node:fs";

const files = [
  ["docs/data/gemma4-q5km-cold-hdd-lazy-pilot.json", false],
  ["docs/data/gemma4-q5km-cold-hdd-reclaimed-pilot.json", true],
];
const expectedModel = "6b4f8074239f72543997a950ff6ae4509553c599f5b432748309ff3438416493";
const expectedLibrary = "5aa19036113383c4b5cf280ed2addf63ffad0318f51913b31e63abcd69c48532";

for (const [file, reclaimed] of files) {
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  if (data.manifest.sha256 !== expectedModel) throw new Error(`${file}: wrong model`);
  if (data.manifest.cpu_library_sha256 !== expectedLibrary) throw new Error(`${file}: wrong patched binary`);
  if (data.manifest.expert_reclaim !== reclaimed) throw new Error(`${file}: wrong policy`);
  if (data.manifest.cache_before_load !== "cold") throw new Error(`${file}: not a cold start`);
  if (data.manifest.warmup !== "off") throw new Error(`${file}: warmup was enabled`);
  if (data.physical_io.startup.residency_before.resident_pages !== 0) {
    throw new Error(`${file}: model was resident before startup`);
  }
  const trial = data.physical_io.trials[0];
  if (trial.server_read_bytes !== 8_119_717_888) throw new Error(`${file}: request byte drift`);
  if (trial.device_read_ios_upper_bound !== 1_982_353) throw new Error(`${file}: I/O count drift`);
  if (trial.device_average_read_bytes_upper_bound !== 4096) throw new Error(`${file}: read size drift`);
  if (data.response_quality.tests_passed !== 1) throw new Error(`${file}: executable task failed`);
}

console.log("cold HDD pilot data: ok");
