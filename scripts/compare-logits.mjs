#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const [referencePath, candidatePath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error("usage: compare-logits.mjs REFERENCE.f32 CANDIDATE.f32 OUTPUT.json");
const reference = fs.readFileSync(referencePath);
const candidate = fs.readFileSync(candidatePath);
if (reference.length !== candidate.length || reference.length % 4 !== 0) {
  throw new Error("logit arrays must have equal lengths divisible by four");
}
let differing = 0;
let maxAbsoluteError = 0;
let sumSquaredError = 0;
let dot = 0;
let referenceSquared = 0;
let candidateSquared = 0;
for (let offset = 0; offset < reference.length; offset += 4) {
  const a = reference.readFloatLE(offset);
  const b = candidate.readFloatLE(offset);
  if (!Object.is(a, b)) differing += 1;
  const error = Math.abs(a - b);
  maxAbsoluteError = Math.max(maxAbsoluteError, error);
  sumSquaredError += error * error;
  dot += a * b;
  referenceSquared += a * a;
  candidateSquared += b * b;
}
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const result = {
  schema: "emufpga.logit-equivalence.v1",
  model: "google_gemma-4-26B-A4B-it-Q5_K_M.gguf",
  model_sha256: "6b4f8074239f72543997a950ff6ae4509553c599f5b432748309ff3438416493",
  llama_cpp_revision: "4d9176092d00586775af140581bb0b558ddc4389",
  comparison: "same patched binary, model, Q5_K_M weights, GPU placement, prompt, and native kernels; resident expert mmap pages versus MADV_RANDOM before and MADV_DONTNEED after each selected expert",
  logits: reference.length / 4,
  bytes_per_array: reference.length,
  differing_float_values: differing,
  max_absolute_error: maxAbsoluteError,
  root_mean_square_error: Math.sqrt(sumSquaredError / (reference.length / 4)),
  cosine_similarity: dot / Math.sqrt(referenceSquared * candidateSquared),
  reference_sha256: sha256(reference),
  candidate_sha256: sha256(candidate),
  caveat: "This isolates the memory-policy change for one complete prompt path; it is not an independent implementation oracle or a coding-quality evaluation."
};
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result));
