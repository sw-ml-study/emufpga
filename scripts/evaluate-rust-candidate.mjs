#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const [responsePath, testsPath, resultPath] = process.argv.slice(2);
if (!resultPath) throw new Error("usage: evaluate-rust-candidate.mjs RESPONSE TESTS RESULT");
const response = fs.readFileSync(responsePath, "utf8").trim();
const tests = fs.readFileSync(testsPath, "utf8").trim();
const fenced = response.match(/```(?:rust)?\s*([\s\S]*?)```/i);
const source = (fenced?.[1] ?? response).trim();
const strict = response === source && /^fn\s+/.test(source);
const directory = fs.mkdtempSync(path.join(os.tmpdir(), "emufpga-rust-eval-"));
try {
  fs.writeFileSync(path.join(directory, "main.rs"), `${source}\nfn main() { ${tests} }\n`);
  fs.chmodSync(directory, 0o777);
  const compile = spawnSync("timeout", ["30", "docker", "run", "--rm", "--network", "none",
    "--read-only", "--memory", "512m", "--cpus", "1", "--pids-limit", "64",
    "--ulimit", "cpu=15:20", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--tmpfs", "/tmp:rw,nosuid,noexec,size=64m",
    "--user", "65534:65534", "--workdir", "/work", "-v", `${directory}:/work:rw`,
    "rust:1.89-slim-bookworm", "rustc", "--edition=2024", "-C", "opt-level=0",
    "-C", "target-feature=+crt-static", "main.rs", "-o", "candidate"],
    { encoding: "utf8", timeout: 35000 });
  let run = null;
  if (compile.status === 0) {
    run = spawnSync("timeout", ["12", "docker", "run", "--rm", "--network", "none",
      "--read-only", "--memory", "128m", "--cpus", "0.5", "--pids-limit", "32",
      "--ulimit", "cpu=5:6", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
      "--user", "65534:65534", "-v", `${directory}:/work:ro`, "ubuntu:24.04", "/work/candidate"],
      { encoding: "utf8", timeout: 15000 });
  }
  const result = { strict, compiled: compile.status === 0, passed: run?.status === 0,
    compile_status: compile.status, run_status: run?.status ?? null,
    compile_stderr: compile.stderr?.slice(0, 2000) ?? "", run_stderr: run?.stderr?.slice(0, 2000) ?? "" };
  fs.writeFileSync(resultPath, JSON.stringify(result) + "\n");
} finally {
  fs.rmSync(directory, { recursive: true, force: true });
}
