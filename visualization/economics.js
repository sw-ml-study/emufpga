"use strict";
(function(root){
  function analyze(x){
    const gib=1073741824, model=x.model_gib*gib, kv=x.context*x.kv_bytes;
    const resident=model+kv, serial=132306+kv, budget=x.budget_gib*gib;
    const union=x.total_experts*(1-Math.pow(1-x.active_experts/x.total_experts,x.batch));
    const expertBytes=model*0.9015, blindBytes=expertBytes/x.batch, unionBytes=expertBytes*(union/x.total_experts)/x.batch;
    const compute=x.compute_tops*1e12/(2*x.active_params_b*1e9);
    const blindBandwidth=x.storage_gbps*1e9/blindBytes, unionBandwidth=x.storage_gbps*1e9/unionBytes;
    let verdict=resident<=budget?"RESIDENT FITS: streaming has no capacity necessity":serial>budget?"NEITHER FITS: KV/state exceeds the budget":"SERIAL FITS; RESIDENT DOES NOT";
    return {resident,serial,budget,union,blind_use:x.active_experts/x.total_experts,union_use:union/x.total_experts,compute,blind:Math.min(compute,blindBandwidth),selected:Math.min(compute,unionBandwidth),block_ns:62*1000/x.clock_mhz,verdict};
  }
  root.EmuEconomics={analyze}; if(typeof module!=="undefined")module.exports={analyze};
})(globalThis);
