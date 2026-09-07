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
const lifetime = JSON.parse(fs.readFileSync("docs/data/gemma4-demand-cache-break-even.json"));
if (lifetime.groups.length !== 6 || lifetime.correctness.policy_digest_mismatches !== 0) throw new Error("break-even analysis incomplete");
const direct = JSON.parse(fs.readFileSync("docs/data/gemma4-direct-cache-break-even.json"));
if (direct.runs.length !== 3 || direct.correctness.digest_mismatches !== 0) throw new Error("direct cache analysis incomplete");
const adapter = JSON.parse(fs.readFileSync("docs/data/gemma4-inference-cache-adapter.json"));
if (!adapter.correct || adapter.max_error > adapter.tolerance) throw new Error("cache adapter arithmetic failed");
if (!adapter.native_span_hook.balanced || adapter.native_span_hook.acquires !== adapter.native_span_hook.releases) throw new Error("native span leases are unbalanced");
if (new Set([adapter.native_span_hook.resident_sha256, adapter.native_span_hook.reclaimed_sha256, adapter.native_span_hook.passthrough_sha256]).size !== 1) throw new Error("native span hook changed logits");
const native = adapter.native_bounded_provider;
if (!native.balanced || native.acquires !== native.releases || native.control_sha256 !== native.candidate_sha256) throw new Error("native bounded provider failed correctness");
if (native.peak_bytes > native.budget_bytes) throw new Error("native bounded provider exceeded budget");
const concurrent = JSON.parse(fs.readFileSync("docs/data/gemma4-concurrent-native-cache-pilot.json"));
if (concurrent.candidate.correct !== concurrent.candidate.requests) throw new Error("native server pilot tasks failed");
if (concurrent.candidate.peak_bytes > 512 * 1024 * 1024) throw new Error("native server cache exceeded budget");
if (!concurrent.oracle_layer_policy.bit_identical) throw new Error("layer-aware provider changed logits");
console.log("model install profiles: structurally valid; Rust tests cover payload checksums");
