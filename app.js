"use strict";

const fallback = Array.from({length: 32}, (_, expert) => ({
  layer: 0, expert, selected: [1,4,8,12,17,21,26,30].includes(expert),
  routed_tokens: [1,4,8,12,17,21,26,30].includes(expert) ? 1 : 0,
  packed_bytes: 1314816, decoded_bytes: [1,4,8,12,17,21,26,30].includes(expert) ? 6291456 : 0,
  layer_read_us: 13869, layer_decode_us: 8667, layer_compute_us: 65547
}));
let events = fallback, frame = 0, playing = true, timer, traffic = 0, useful = 0;
let routingAnalysis = null;
const $ = id => document.getElementById(id);
const benchmark = [[1,42.21,10.65,13.0,12.9,8],[4,42.21,25.12,18.2,18.0,19],[8,42.21,36.95,19.4,18.5,28],[16,42.21,36.95,21.5,22.2,28],[32,42.21,36.95,26.1,24.9,28]];
const binary = bytes => bytes >= 1073741824 ? `${(bytes/1073741824).toFixed(2)} GiB` : bytes >= 1048576 ? `${(bytes/1048576).toFixed(1)} MiB` : `${Math.round(bytes/1024)} KiB`;

function buildAnalysis() {
  $("bench").innerHTML=benchmark.map(r=>`<tr><td>${r[0]}</td><td>${r[1].toFixed(2)} MB</td><td>${r[2].toFixed(2)} MB</td><td>${r[3].toFixed(1)}</td><td>${r[4].toFixed(1)}</td><td>${r[5]}/32</td></tr>`).join("");
  const bars=(items,max)=>items.map(i=>`<div class="bar-row"><span>${i[0]}</span><div class="bar-track"><div class="bar ${i[3]}" style="width:${100*i[1]/max}%"></div></div><b>${i[2]}</b></div>`).join("");
  $("traffic-chart").innerHTML=benchmark.flatMap(r=>[[`B${r[0]} all`,r[1],`${r[1].toFixed(1)} MB`,"resident"],[`B${r[0]} union`,r[2],`${r[2].toFixed(1)} MB`,"serial"]]).map(i=>bars([i],42.21)).join("");
  const points=(column)=>benchmark.map((r,i)=>`${42+i*88},${175-(r[column]-10)*7}`).join(" ");
  $("throughput-chart").innerHTML=`<path class="chart-grid" d="M38 20V175H430M38 105H430M38 35H430"/><polyline class="chart-line line-resident" points="${points(3)}"/><polyline class="chart-line line-serial" points="${points(4)}"/>${benchmark.map((r,i)=>`<text class="chart-label" x="${34+i*88}" y="195">B${r[0]}</text>`).join("")}<text class="chart-label" x="48" y="32">amber all-expert · cyan selected-union · tok/s</text>`;
  const update=()=>{const kv=Number($("context").value)*49152,resident=1099212096+kv,serial=132306+kv;$("resident-memory").textContent=`${binary(resident)} minimum`;$("serial-memory").textContent=`${binary(kv)} KV + 129 KiB`;$("union-memory").textContent=`${binary(kv)} KV + 129 KiB`;$("memory-chart").innerHTML=bars([["resident",resident,binary(resident),"resident"],["serial",serial,binary(serial),"serial"]],resident);};
  $("context").onchange=update; update();
  document.querySelectorAll(".knobs input").forEach(input=>input.oninput=renderLab); renderLab();
}

function renderLab() {
  const n=id=>Number($(id).value), x={model_gib:n("model-gib"),budget_gib:n("budget-gib"),total_experts:n("total-experts"),active_experts:n("active-experts"),batch:n("batch"),context:n("lab-context"),kv_bytes:n("kv-bytes"),storage_gbps:n("storage-gbps"),compute_tops:n("compute-tops"),active_params_b:n("active-params"),clock_mhz:n("clock-mhz")};
  if(x.active_experts>x.total_experts){$("warnings").textContent="INVALID: active experts cannot exceed total experts.";return;}
  const empirical=routingAnalysis&&x.total_experts===32&&x.active_experts===8?routingAnalysis.union_curve.find(row=>row.batch===x.batch):null;if(empirical)x.union_experts=empirical.observed_mean;
  const r=EmuEconomics.analyze(x), max=Math.max(r.resident,r.serial,r.budget), row=(name,value,label,kind)=>`<div class="bar-row"><span>${name}</span><div class="bar-track"><div class="bar ${kind}" style="width:${100*value/max}%"></div></div><b>${label}</b></div>`;
  $("decision").className=`decision ${r.verdict.startsWith("SERIAL")?"good":"caution"}`;$("decision").textContent=r.verdict;
  $("fit-graph").innerHTML=row("budget",r.budget,binary(r.budget),"budget")+row("resident",r.resident,binary(r.resident),"resident")+row("serial",r.serial,binary(r.serial),"serial");
  const ceiling=Math.max(r.compute,r.blind,r.selected), meter=(name,value)=>`<div class="bar-row"><span>${name}</span><meter min="0" max="${ceiling}" value="${value}"></meter><b>${value.toFixed(1)} tok/s</b></div>`;$("ceiling-graph").innerHTML=meter("compute",r.compute)+meter("blind",r.blind)+meter("union",r.selected);
  $("usefulness-graph").innerHTML=`<div class="donut" style="--value:${r.blind_use}"><b>${(100*r.blind_use).toFixed(0)}%</b><span>blind useful</span></div><div class="donut" style="--value:${r.union_use}"><b>${(100*r.union_use).toFixed(0)}%</b><span>union fetched</span></div>`;
  const routeBasis=empirical?`MEASURED ROUTING: B${x.batch} averages ${empirical.observed_mean.toFixed(1)} experts across ${empirical.samples} prompt/layer samples; tiny prompt-prefix corpus.`:"PROJECTED ROUTING: uniform-independent formula; no matching empirical Granite observation.";const warnings=[routeBasis,"PROJECTED THROUGHPUT: assumes 90.15% of model bytes are expert weights.",`HYPOTHETICAL: 62 block cycles at ${x.clock_mhz} MHz = ${r.block_ns.toFixed(0)} ns; this is not end-to-end latency.`];if(r.resident<=r.budget)warnings.unshift("CAUTION: the model already fits; streaming may add latency without solving a capacity problem.");if(r.blind_use<.5)warnings.unshift("CAUTION: blind streaming discards most expert bandwidth.");if(r.serial>r.budget)warnings.unshift("STOP: streaming weights cannot solve a KV/state capacity overflow.");$("warnings").innerHTML=warnings.map(w=>`<p>${w}</p>`).join("");
}

function renderRoutingAnalysis(data){routingAnalysis=data;const bars=(items,max)=>items.map(i=>`<div class="bar-row"><span>${i[0]}</span><div class="bar-track"><div class="bar ${i[3]}" style="width:${100*i[1]/max}%"></div></div><b>${i[2]}</b></div>`).join("");$("traffic-chart").innerHTML=data.union_curve.flatMap(r=>[[`B${r.batch} observed`,r.observed_mean,`${r.observed_mean.toFixed(1)}/32`,"serial"],[`B${r.batch} independent`,r.independent_mean,`${r.independent_mean.toFixed(1)}/32`,"resident"]]).map(i=>bars([i],32)).join("");const last=data.union_curve.at(-1);$("routing-takeaway").textContent=`At B${last.batch}, observed routes touch ${last.observed_mean.toFixed(1)}/32 experts versus ${last.independent_mean.toFixed(1)} predicted. This may reflect route correlation, but the corpus has only ${data.totals.tokens} prompt tokens.`;renderLab();}

function renderTiming(data){const all=data.groups.filter(g=>g.schedule==="all-expert"),selected=data.groups.filter(g=>g.schedule==="selected-union"),max=Math.max(...data.groups.map(g=>g.tokens_s.max)),point=(g,i)=>`${60+i*150},${175-g.tokens_s.p50/max*140}`;$("throughput-chart").innerHTML=`<path class="chart-grid" d="M38 20V175H430M38 105H430M38 35H430"/><polyline class="chart-line line-resident" points="${all.map(point).join(" ")}"/><polyline class="chart-line line-serial" points="${selected.map(point).join(" ")}"/>${all.map((g,i)=>`<text class="chart-label" x="${48+i*150}" y="195">B${g.batch}</text>`).join("")}<text class="chart-label" x="48" y="32">amber all-expert · cyan selected-union · median tok/s</text>`;$("timing-takeaway").textContent=`B1 median: ${all[0].tokens_s.p50.toFixed(1)} all-expert versus ${selected[0].tokens_s.p50.toFixed(1)} selected-union tok/s. At B8: ${all[2].tokens_s.p50.toFixed(1)} versus ${selected[2].tokens_s.p50.toFixed(1)}; scalar compute dominates.`;$("bench").innerHTML=all.map((g,i)=>{const s=selected[i];return `<tr><td>${g.batch}</td><td>${g.tokens_s.p50.toFixed(1)} (${g.tokens_s.min.toFixed(1)}–${g.tokens_s.max.toFixed(1)})</td><td>${s.tokens_s.p50.toFixed(1)} (${s.tokens_s.min.toFixed(1)}–${s.tokens_s.max.toFixed(1)})</td><td colspan="3">7 warm runs</td></tr>`}).join("");}

function renderStorage(data){const rows=data.groups.filter(g=>g.mode==="direct"),max=Math.max(...rows.map(g=>g.gb_s.p90));$("storage-chart").innerHTML=rows.map(g=>`<div class="bar-row storage-row"><span>${g.tier}</span><div class="bar-track"><div class="bar ${g.tier.includes("selected")?"serial":"resident"}" style="width:${100*g.gb_s.p50/max}%"></div></div><b>${g.gb_s.p50.toFixed(2)} <small>(${g.gb_s.p10.toFixed(2)}–${g.gb_s.p90.toFixed(2)}) GB/s</small></b></div>`).join("");const hdd=data.overlap.find(x=>x.tier==="hdd-all"&&x.mode==="direct"),nvme=data.overlap.find(x=>x.tier==="nvme-all"&&x.mode==="direct");$("storage-takeaway").textContent=`Cache-bypass median: HDD ${rows.find(x=>x.tier==="hdd-all").gb_s.p50.toFixed(2)} GB/s; NVMe ${rows.find(x=>x.tier==="nvme-all").gb_s.p50.toFixed(2)} GB/s. Idealized B1 overlap ceiling: ${hdd.maximum_saving_percent.toFixed(0)}% and ${nvme.maximum_saving_percent.toFixed(0)}%; this is not implemented runtime speedup.`;}

function renderPrefetch(data){const rows=data.groups.filter(g=>g.backend==="prefetch");$("prefetch-chart").innerHTML=rows.map(g=>{const v=g.speedup_percent,side=v>=0?"gain":"loss";return `<div class="delta-row"><span>${g.tier} ${g.schedule}<small>${g.cache}</small></span><div class="delta-track"><i class="${side}" style="width:${Math.min(50,Math.abs(v)*3)}%;${v>=0?"left:50%":"right:50%"}"></i></div><b class="${side}">${v>=0?"+":""}${v.toFixed(1)}%</b></div>`}).join("");const values=rows.map(x=>x.speedup_percent);$("prefetch-takeaway").textContent=`Observed range ${Math.min(...values).toFixed(1)}% to +${Math.max(...values).toFixed(1)}%. Cache-bypass cases gained 2–5%; warm cases were noisy/mixed. Correctness stayed within the 0.002 gate.`;}

function buildBoard() {
  for (let i=0;i<32;i++) {
    const x=(i%8)*82, y=Math.floor(i/8)*54+22;
    $("experts").insertAdjacentHTML("beforeend", `<rect id="e${i}" class="expert" x="${x}" y="${y}" width="68" height="38" rx="6"/><text class="expert-label" x="${x+17}" y="${y+24}">E${String(i).padStart(2,"0")}</text>`);
  }
  for(let i=0;i<8;i++) $("lanes").insertAdjacentHTML("beforeend", `<rect class="lane" x="${18+i*20}" y="58" width="13" height="52" rx="4"/>`);
  for(let i=0;i<5;i++) $("fifo-bars").insertAdjacentHTML("beforeend", `<rect class="fifo-bar" x="18" y="${55+i*14}" width="${105-i*9}" height="8" rx="3"/>`);
}

function renderArchitecture(id) {
  const scenario=EmuArchitectures.scenarios[id],nodes=EmuArchitectures.nodes,active=new Set(scenario.active),centers={};
  for(const [key,[x,y]] of Object.entries(nodes)) centers[key]=[x+65,y+32];
  $("architecture-edges").innerHTML=scenario.edges.map((e,i)=>{const [x1,y1]=centers[e.from],[x2,y2]=centers[e.to],mx=(x1+x2)/2,my=(y1+y2)/2;return `<g><path id="flow-${i}" class="flow-edge ${e.kind}" d="M${x1} ${y1}L${x2} ${y2}" marker-end="url(#arrow-${e.kind})"/><circle class="flow-particle ${e.kind}" r="5"><animateMotion dur="${1.4+(i%3)*.35}s" repeatCount="indefinite"><mpath href="#flow-${i}"/></animateMotion></circle><text class="flow-label" x="${mx}" y="${my-7}">${e.label}</text></g>`}).join("");
  $("architecture-nodes").innerHTML=Object.entries(nodes).map(([key,[x,y,title,sub]])=>`<g class="flow-node ${active.has(key)?"active":"inactive"}" transform="translate(${x} ${y})"><rect width="130" height="64" rx="10"/><text x="65" y="27">${title}</text><text class="sub" x="65" y="47">${sub}</text></g>`).join("");
  $("architecture-thesis").textContent=scenario.thesis;$("architecture-state").textContent=scenario.state;
}

function visibleEvents() {
  const mode=$("schedule").value;
  return mode === "union" ? events.filter(e=>e.selected) : events;
}

function render() {
  const list=visibleEvents(); if(!list.length) return;
  frame%=list.length; const e=list[frame], mode=$("schedule").value;
  document.querySelectorAll(".expert").forEach(n=>n.classList.remove("selected","current"));
  events.filter(x=>x.layer===e.layer&&x.selected).forEach(x=>$("e"+x.expert).classList.add("selected"));
  $("e"+e.expert).classList.add("current");
  document.querySelectorAll(".lane").forEach(n=>n.classList.toggle("on",e.selected));
  document.querySelectorAll(".fifo-bar").forEach((n,i)=>n.classList.toggle("on",i<3&&!e.selected&&mode==="serial"));
  const packet=$("packet"), path=[210,270,415,475,640,700,890,950]; packet.setAttribute("cx",path[frame%path.length]);
  const moved=mode==="resident"?0:e.packed_bytes; traffic+=moved; if(e.selected) useful+=e.packed_bytes;
  $("event-title").textContent=`Layer ${e.layer} • Expert ${e.expert}`;
  $("event-state").textContent=e.selected?"ENABLED: decode + MAC":"DISABLED: drain only";
  $("event-state").setAttribute("fill",e.selected?"#68f7a4":"#7fa9ad");
  $("event-route").textContent=`routed tokens: ${e.routed_tokens}`;
  $("event-bytes").textContent=`packed expert: ${e.packed_bytes.toLocaleString()} B`;
  $("event-timing").textContent=`layer: ${e.layer_read_us/1000} / ${e.layer_decode_us/1000} / ${e.layer_compute_us/1000} ms`;
  $("scope").textContent=mode==="serial"?"MEASURED layer totals":"PROJECTED schedule using measured bytes";
  $("position").textContent=`${e.layer} / ${e.expert}`; $("traffic").textContent=traffic.toLocaleString(); $("useful").textContent=useful.toLocaleString();
  $("utilization").textContent=traffic?`${(100*useful/traffic).toFixed(1)}%`:"resident";
}

function restart() { clearInterval(timer); const delay=1100-$("speed").value*100; timer=setInterval(()=>{if(playing){frame++;render();}},delay); }
$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play"};
$("step").onclick=()=>{frame++;render()}; $("speed").oninput=restart;
$("schedule").onchange=()=>{frame=traffic=useful=0;render()};

buildBoard(); buildAnalysis(); renderArchitecture($("architecture-select").value);$("architecture-select").onchange=e=>renderArchitecture(e.target.value);fetch("trace.json").then(r=>r.ok?r.json():Promise.reject()).then(t=>{events=t.events;render()}).catch(()=>render()); restart();
fetch("build-info.json").then(r=>r.ok?r.json():Promise.reject()).then(b=>{$("build-info").textContent=`built on ${b.host} · ${b.sha} · ${b.timestamp}`}).catch(()=>{});
fetch("routing-analysis.json").then(r=>r.ok?r.json():Promise.reject()).then(renderRoutingAnalysis).catch(()=>{});
fetch("granite-timing.json").then(r=>r.ok?r.json():Promise.reject()).then(renderTiming).catch(()=>{});
fetch("storage-tier-analysis.json").then(r=>r.ok?r.json():Promise.reject()).then(renderStorage).catch(()=>{});
fetch("prefetch-analysis.json").then(r=>r.ok?r.json():Promise.reject()).then(renderPrefetch).catch(()=>{});
