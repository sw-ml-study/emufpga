const assert=require("node:assert/strict");const {analyze}=require("./economics.js");
const base={model_gib:1.024,budget_gib:16,total_experts:32,active_experts:8,batch:1,context:32768,kv_bytes:49152,storage_gbps:3,compute_tops:1,active_params_b:.4,clock_mhz:100};
const a=analyze(base);assert.match(a.verdict,/RESIDENT FITS/);assert.equal(a.blind_use,.25);assert.ok(a.selected>=a.blind);assert.equal(a.block_ns,620);
const large=analyze({...base,model_gib:300});assert.match(large.verdict,/SERIAL FITS/);const kv=analyze({...base,context:1e7});assert.match(kv.verdict,/NEITHER FITS/);
const batched=analyze({...base,batch:32});assert.ok(batched.union>8);const empirical=analyze({...base,batch:4,union_experts:18.3125});assert.equal(empirical.union,18.3125);assert.ok(empirical.selected>analyze({...base,batch:4}).selected);console.log("economics calculations: ok");
