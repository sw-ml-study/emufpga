"use strict";

const fallback = Array.from({length: 32}, (_, expert) => ({
  layer: 0, expert, selected: [1,4,8,12,17,21,26,30].includes(expert),
  routed_tokens: [1,4,8,12,17,21,26,30].includes(expert) ? 1 : 0,
  packed_bytes: 1314816, decoded_bytes: [1,4,8,12,17,21,26,30].includes(expert) ? 6291456 : 0,
  layer_read_us: 13869, layer_decode_us: 8667, layer_compute_us: 65547
}));
let events = fallback, frame = 0, playing = true, timer, traffic = 0, useful = 0;
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
}

function buildBoard() {
  for (let i=0;i<32;i++) {
    const x=(i%8)*82, y=Math.floor(i/8)*54+22;
    $("experts").insertAdjacentHTML("beforeend", `<rect id="e${i}" class="expert" x="${x}" y="${y}" width="68" height="38" rx="6"/><text class="expert-label" x="${x+17}" y="${y+24}">E${String(i).padStart(2,"0")}</text>`);
  }
  for(let i=0;i<8;i++) $("lanes").insertAdjacentHTML("beforeend", `<rect class="lane" x="${18+i*20}" y="58" width="13" height="52" rx="4"/>`);
  for(let i=0;i<5;i++) $("fifo-bars").insertAdjacentHTML("beforeend", `<rect class="fifo-bar" x="18" y="${55+i*14}" width="${105-i*9}" height="8" rx="3"/>`);
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

buildBoard(); buildAnalysis(); fetch("trace.json").then(r=>r.ok?r.json():Promise.reject()).then(t=>{events=t.events;render()}).catch(()=>render()); restart();
fetch("build-info.json").then(r=>r.ok?r.json():Promise.reject()).then(b=>{$("build-info").textContent=`built on ${b.host} · ${b.sha} · ${b.timestamp}`}).catch(()=>{});
