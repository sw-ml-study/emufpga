#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const [input, output] = process.argv.slice(2);
if (!output) throw new Error("usage: analyze-gemma4-offload.mjs INPUT_DIR OUTPUT.json");
const manifest = JSON.parse(fs.readFileSync(path.join(input, "manifest.json"), "utf8"));
const summaries = fs.readFileSync(path.join(input, "summaries.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
const taskPath = path.resolve("docs/data", manifest.tasks ?? "gemma4-correctness-tasks.json");
const tasks = fs.existsSync(taskPath) ? JSON.parse(fs.readFileSync(taskPath, "utf8")) : [];
const telemetryLines = fs.readFileSync(path.join(input, "telemetry.csv"), "utf8").trim().split("\n");
const detailedTelemetry = telemetryLines[0].startsWith("timestamp_s,");
const telemetry = telemetryLines.slice(detailedTelemetry ? 1 : 0).map((line) => {
  const values = line.split(",").map(Number);
  if (!detailedTelemetry) {
    const [seconds, rssKib, vramMib, utilization, powerWatts] = values;
    return { seconds, rssKib, vramMib, utilization, powerWatts };
  }
  const [seconds, rssKib, rssAnonKib, rssFileKib, pssKib, pssAnonKib, pssFileKib,
    readBytes, rchar, vramMib, utilization, powerWatts] = values;
  return { seconds, rssKib, rssAnonKib, rssFileKib, pssKib, pssAnonKib, pssFileKib,
    readBytes, rchar, vramMib, utilization, powerWatts };
});
const phasesPath = path.join(input, "phases.csv");
const phaseRows = fs.existsSync(phasesPath)
  ? fs.readFileSync(phasesPath, "utf8").trim().split("\n").slice(1).map((line) => {
      const [seconds, concurrency, event] = line.split(",");
      return { seconds: Number(seconds), concurrency: Number(concurrency), event };
    })
  : [];
const phases = phaseRows.filter((item) => item.event === "start").map((start) => ({
  concurrency: start.concurrency,
  start: start.seconds,
  end: phaseRows.find((item) => item.event === "end" && item.concurrency === start.concurrency)?.seconds,
})).filter((item) => item.end !== undefined);
const requestRecords = summaries.flatMap((summary) => {
  const requestPath = path.join(input, `requests-${summary.concurrency}.jsonl`);
  return fs.existsSync(requestPath)
    ? fs.readFileSync(requestPath, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse)
    : [];
});
const snapshotsPath = path.join(input, "io-snapshots.jsonl");
const snapshots = fs.existsSync(snapshotsPath)
  ? fs.readFileSync(snapshotsPath, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse)
  : [];
const startupPath = path.join(input, "startup-io.json");
const startupIo = fs.existsSync(startupPath)
  ? JSON.parse(fs.readFileSync(startupPath, "utf8"))
  : null;
const derivedStartupIo = startupIo === null ? null : {
  ...startupIo,
  device_read_bytes_upper_bound: startupIo.device_read_sectors_delta * startupIo.sector_bytes,
};
const gradedRequests = requestRecords.map((record) => {
  const task = tasks[record.request_id - 1];
  if (typeof record.passed === "boolean") return record;
  const accepted = task === undefined ? [record.expected] : [task.expected, ...(task.accepted ?? [])];
  return {
    ...record,
    answer_present: accepted.some((answer) => record.content.toLocaleLowerCase().includes(answer.toLocaleLowerCase())),
  };
});
const serverLog = fs.readFileSync(path.join(input, "server.log"), "utf8");
const expertOps = [...serverLog.matchAll(/^spm_mmid (?:timestamp_s=([0-9.]+) )?.*activations=(\d+) assignments=(\d+) selected=(\d+) expert_bytes=(\d+) logical_bytes=(\d+)$/gm)].map((match) => ({
  seconds: match[1] === undefined ? null : Number(match[1]),
  activations: Number(match[2]), assignments: Number(match[3]), selected: Number(match[4]),
  expertBytes: Number(match[5]), logicalBytes: Number(match[6]),
}));
const ioTrials = snapshots.filter(item => item.event === "start").map((start) => {
  const end = snapshots.find(item => item.event === "end" &&
    item.concurrency === start.concurrency && item.run === start.run);
  if (!end) throw new Error(`missing I/O end snapshot for concurrency ${start.concurrency} run ${start.run}`);
  const records = gradedRequests.filter(item => item.concurrency === start.concurrency && item.run === start.run);
  const generatedTokens = records.reduce((sum, item) => sum + (item.output_tokens ?? 0), 0);
  const serverReadBytes = end.server_read_bytes - start.server_read_bytes;
  const deviceReadIos = end.device_read_ios - start.device_read_ios;
  const deviceReadBytes = (end.device_read_sectors - start.device_read_sectors) * 512;
  const timedOps = expertOps.filter(item => item.seconds !== null &&
    item.seconds >= start.timestamp_ms / 1000 && item.seconds <= end.timestamp_ms / 1000);
  return {
    concurrency: start.concurrency,
    run: start.run,
    cache_mode: start.cache_mode,
    elapsed_ms: end.timestamp_ms - start.timestamp_ms,
    request_wall_ms_max: Math.max(...records.map(item => item.wall_ms)),
    generated_tokens: generatedTokens,
    passed: records.filter(item => item.passed).length,
    server_read_bytes: serverReadBytes,
    device_read_ios_upper_bound: deviceReadIos,
    device_read_bytes_upper_bound: deviceReadBytes,
    device_average_read_bytes_upper_bound: deviceReadIos === 0 ? null : deviceReadBytes / deviceReadIos,
    server_read_bytes_per_token: generatedTokens === 0 ? null : serverReadBytes / generatedTokens,
    device_read_bytes_per_token_upper_bound: generatedTokens === 0 ? null : deviceReadBytes / generatedTokens,
    minor_faults: end.minor_faults - start.minor_faults,
    major_faults: end.major_faults - start.major_faults,
    resident_bytes_start: start.residency?.resident_bytes ?? null,
    resident_bytes_end: end.residency?.resident_bytes ?? null,
    expert_trace: summarizeExpertOps(timedOps),
  };
});
function summarizeSamples(samples) {
  if (samples.length === 0) return null;
  const result = {
    samples: samples.length,
    peak_process_rss_mib: Math.max(...samples.map((item) => item.rssKib)) / 1024,
    peak_vram_mib: Math.max(...samples.map((item) => item.vramMib)),
  };
  if (detailedTelemetry) {
    result.peak_rss_anon_mib = Math.max(...samples.map((item) => item.rssAnonKib)) / 1024;
    result.peak_rss_file_mib = Math.max(...samples.map((item) => item.rssFileKib)) / 1024;
    result.peak_pss_mib = Math.max(...samples.map((item) => item.pssKib)) / 1024;
    result.peak_pss_anon_mib = Math.max(...samples.map((item) => item.pssAnonKib)) / 1024;
    result.peak_pss_file_mib = Math.max(...samples.map((item) => item.pssFileKib)) / 1024;
    result.process_read_bytes_delta = Math.max(...samples.map((item) => item.readBytes)) - Math.min(...samples.map((item) => item.readBytes));
    result.process_rchar_delta = Math.max(...samples.map((item) => item.rchar)) - Math.min(...samples.map((item) => item.rchar));
  }
  return result;
}
function summarizeExpertOps(ops) {
  if (ops.length === 0) return null;
  return {
    operations: ops.length,
    activations: ops.reduce((sum, item) => sum + item.activations, 0),
    assignments: ops.reduce((sum, item) => sum + item.assignments, 0),
    selected_experts: ops.reduce((sum, item) => sum + item.selected, 0),
    logical_bytes: ops.reduce((sum, item) => sum + item.logicalBytes, 0),
  };
}
let gpuEnergyJoules = 0;
for (let index = 1; index < telemetry.length; index += 1) {
  const previous = telemetry[index - 1];
  const current = telemetry[index];
  gpuEnergyJoules += (current.seconds - previous.seconds) * (current.powerWatts + previous.powerWatts) / 2;
}
const result = {
  schema: "emufpga.gemma4-offload-derived.v1",
  captured_utc: new Date().toISOString(),
  manifest,
  telemetry: {
    scope: `server load plus complete ${summaries.map((item) => item.concurrency).join("/")} request sweep; NVIDIA board energy only`,
    samples: telemetry.length,
    elapsed_seconds: telemetry.at(-1).seconds - telemetry[0].seconds,
    peak_process_rss_mib: Math.max(...telemetry.map((item) => item.rssKib)) / 1024,
    mean_process_rss_mib: telemetry.reduce((sum, item) => sum + item.rssKib, 0) / telemetry.length / 1024,
    minimum_process_rss_mib: Math.min(...telemetry.map((item) => item.rssKib)) / 1024,
    peak_vram_mib: Math.max(...telemetry.map((item) => item.vramMib)),
    peak_gpu_utilization_percent: Math.max(...telemetry.map((item) => item.utilization)),
    gpu_board_energy_joules: gpuEnergyJoules,
    ...(detailedTelemetry ? summarizeSamples(telemetry) : {}),
    by_concurrency: phases.map((phase) => ({
      concurrency: phase.concurrency,
      ...summarizeSamples(telemetry.filter((item) => item.seconds >= phase.start && item.seconds <= phase.end)),
    })),
  },
  requests: summaries,
  response_quality: gradedRequests.length === 0 ? null : {
    scope: manifest.client === "bench-gemma4-rust-code.mjs"
      ? "dependency-free Rust functions compiled and executed against deterministic tests in an isolated container"
      : "small deterministic smoke corpus; not an agentic coding benchmark",
    responses: gradedRequests.length,
    strict_instruction_following: gradedRequests.filter((item) => item.strict ?? item.correct).length,
    ...(manifest.client === "bench-gemma4-rust-code.mjs" ? {
      compiled: gradedRequests.filter((item) => item.compiled).length,
      tests_passed: gradedRequests.filter((item) => item.passed).length,
    } : { expected_answer_present: gradedRequests.filter((item) => item.answer_present).length }),
    by_concurrency: summaries.map((summary) => {
      const records = gradedRequests.filter((item) => item.concurrency === summary.concurrency);
      return {
        concurrency: summary.concurrency,
        responses: records.length,
        strict_instruction_following: records.filter((item) => item.strict ?? item.correct).length,
        ...(manifest.client === "bench-gemma4-rust-code.mjs" ? {
          compiled: records.filter((item) => item.compiled).length,
          tests_passed: records.filter((item) => item.passed).length,
        } : { expected_answer_present: records.filter((item) => item.answer_present).length }),
      };
    }),
  },
  expert_trace: expertOps.length === 0 ? null : {
    scope: "logical selected-expert tensor bytes requested; not physical storage IO",
    ...summarizeExpertOps(expertOps),
    by_concurrency: phases.map((phase) => ({
      concurrency: phase.concurrency,
      ...summarizeExpertOps(expertOps.filter((item) => item.seconds >= phase.start && item.seconds <= phase.end)),
    })),
  },
  physical_io: snapshots.length === 0 ? null : {
    scope: "llama-server /proc I/O attribution plus whole model-partition sector deltas around inference only",
    model_source: manifest.model_source,
    model_device_stat: manifest.model_device_stat,
    sector_bytes: 512,
    startup: derivedStartupIo,
    trials: ioTrials,
  },
  caveats: [
    manifest.client === "bench-gemma4-rust-code.mjs"
      ? "Eight dependency-free functions are bounded executable evidence, not a repository-scale coding-agent benchmark."
      : "Simple deterministic correctness probes are smoke tests, not a benchmark of model quality.",
    ...(manifest.client === "bench-gemma4-rust-code.mjs" ? [
      "The telemetry phase includes sequential isolated compilation and test execution after each inference group; inference wall time is recorded separately.",
    ] : [
      "Expected-answer containment tolerates explanatory wrappers and is weaker than executable semantic validation.",
      "TTFT is client-observed; inter-token intervals are timestamps of streamed content events.",
    ]),
    "Zero process read_bytes can mean mmap faults were served from warm page cache; it does not prove zero memory traffic or zero prior device IO.",
    "Energy excludes CPU, DRAM, storage, motherboard, fans, and PSU losses.",
    ...(snapshots.length === 0 ? [] : [
      "Whole-device sector deltas are upper bounds because the model device is shared; server /proc read_bytes is the attributable counter.",
      "Per-file POSIX_FADV_DONTNEED is used for cold trials; global drop_caches is never used.",
    ]),
    manifest.expert_reclaim
      ? "Experimental Linux mmap pages are reclaimed after native selected-expert operations; this is not an upstream llama.cpp feature."
      : "Selected expert mappings are left resident as the control memory policy.",
  ],
};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result.telemetry));
