#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const [directory, tasksPath] = process.argv.slice(2);
if (!tasksPath) throw new Error("usage: regrade-gemma4-code.mjs RESULTS_DIR TASKS.json");
const tasks = JSON.parse(fs.readFileSync(tasksPath, "utf8"));
for (const name of fs.readdirSync(directory).filter(item => /^requests-\d+\.jsonl$/.test(item)).sort()) {
  const input = path.join(directory, name);
  const records = fs.readFileSync(input, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
  const updated = records.map((record) => {
    const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "emufpga-regrade-"));
    try {
      const response = path.join(scratch, "response.txt");
      const tests = path.join(scratch, "tests.txt");
      const result = path.join(scratch, "result.json");
      fs.writeFileSync(response, record.content);
      fs.writeFileSync(tests, tasks[record.request_id - 1].tests);
      const run = spawnSync("node", ["scripts/evaluate-rust-candidate.mjs", response, tests, result],
        { encoding: "utf8", timeout: 45000 });
      if (run.status !== 0) throw new Error(`evaluator failed for ${name}: ${run.stderr}`);
      return { ...record, ...JSON.parse(fs.readFileSync(result, "utf8")) };
    } finally { fs.rmSync(scratch, { recursive: true, force: true }); }
  });
  fs.writeFileSync(input, updated.map(JSON.stringify).join("\n") + "\n");
  console.log(`${name}: ${updated.filter(item => item.passed).length}/${updated.length} passed`);
}
