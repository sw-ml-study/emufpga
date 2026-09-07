import fs from "node:fs";
function check(path, expectedState) {
  const envelope = JSON.parse(fs.readFileSync(path));
  if (envelope.schema !== "emufpga.install-manifest-envelope.v1") throw new Error(`${path}: schema`);
  if (!/^[0-9a-f]{64}$/.test(envelope.payload_sha256)) throw new Error(`${path}: checksum format`);
  if (envelope.payload.workload.state !== expectedState) throw new Error(`${path}: state`);
  if (!envelope.payload.inventory.tensors.length) throw new Error(`${path}: inventory`);
}

check("docs/data/granite-install-profile.json", "cold_fallback");
check("docs/data/gemma4-install-profile.json", "calibrated");
check("docs/data/gemma4-install-calibration-profile.json", "calibrated");
const cache = JSON.parse(fs.readFileSync("docs/data/gemma4-profile-cache-analysis.json"));
if (cache.groups.length !== 16 || cache.correctness.mismatches !== 0) throw new Error("profile cache analysis incomplete");
console.log("model install profiles: structurally valid; Rust tests cover payload checksums");
