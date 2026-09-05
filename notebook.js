(function (root, factory) {
  const api = factory(root && root.document);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SPMNotebook = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function (document) {
  "use strict";

  const LESSONS = {
    blind: { title: "Read everything, use only the routed experts.", requests: [[2, 6]], moved: [0, 1, 2, 3, 4, 5, 6, 7], applications: 2, note: "Correct, but 6 of 8 transfers do no useful work." },
    route: { title: "Use the router before moving weights.", requests: [[2, 6]], moved: [2, 6], applications: 2, note: "Traffic falls 4× in this toy layer; routing metadata must arrive first." },
    reuse: { title: "Hold each selected expert long enough to serve both requests.", requests: [[2, 6], [2, 5]], moved: [2, 5, 6], applications: 4, note: "E2 crosses once and serves twice. Batching buys reuse, but adds scheduling latency." }
  };

  function renderLesson(name) {
    const lesson = LESSONS[name] || LESSONS.blind;
    const copy = document.getElementById("lesson-copy");
    const tape = document.getElementById("expert-tape");
    const lines = document.getElementById("request-lines");
    const counts = document.getElementById("lesson-counts");
    if (!copy || !tape || !lines || !counts) return;
    copy.innerHTML = `<h2>${lesson.title}</h2><p>${lesson.note}</p>`;
    lines.innerHTML = lesson.requests.map((needs, index) => `<span><b>request ${String.fromCharCode(65 + index)}</b> router selects ${needs.map(n => `E${n}`).join(" + ")}</span>`).join("");
    tape.innerHTML = Array.from({ length: 8 }, (_, n) => `<i class="${lesson.moved.includes(n) ? "moving" : "parked"}${lesson.requests.some(r => r.includes(n)) ? " selected" : ""}" style="--order:${lesson.moved.indexOf(n)}">E${n}</i>`).join("");
    const efficiency = Math.round(lesson.applications / lesson.moved.length * 100);
    counts.innerHTML = `<span><b>${lesson.moved.length}</b> expert blocks fetched</span><span><b>${lesson.applications}</b> useful applications</span><span><b>${efficiency}%</b> applications per fetch</span>`;
  }

  function selectLesson(button) {
    document.querySelectorAll("[data-lesson]").forEach(b => {
      const active = b === button;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
    });
    renderLesson(button.dataset.lesson);
  }

  function selectEvidence(button) {
    document.querySelectorAll("[data-evidence]").forEach(b => {
      const active = b === button;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll("[data-panel]").forEach(panel => { panel.hidden = panel.dataset.panel !== button.dataset.evidence; });
  }

  async function loadBuildInfo() {
    const target = document.getElementById("build-info");
    if (!target) return;
    try {
      const response = await fetch("build-info.json", { cache: "no-store" });
      if (!response.ok) return;
      const info = await response.json();
      target.textContent = `${info.host} · ${info.sha} · ${info.timestamp}`;
    } catch (_) { /* local file preview keeps the fallback */ }
  }

  function summarizePlacement(data, parallel) {
    const row = placement => data.throughput.find(item => item.placement === placement && item.parallel === parallel);
    const telemetry = placement => data.telemetry.find(item => item.placement === placement).across_runs;
    return {
      gpu: { time: row("all-gpu").total_seconds.p50, generation: row("all-gpu").generation_aggregate_tps.p50, vram: telemetry("all-gpu").peak_gpu_memory_mib.p50, energy: telemetry("all-gpu").gpu_energy_j.p50 },
      cpu: { time: row("experts-cpu").total_seconds.p50, generation: row("experts-cpu").generation_aggregate_tps.p50, vram: telemetry("experts-cpu").peak_gpu_memory_mib.p50, energy: telemetry("experts-cpu").gpu_energy_j.p50 }
    };
  }

  function placementBar(label, value, maximum, unit, kind) {
    const width = Math.max(2, 100 * value / maximum);
    return `<div class="metric-row ${kind}"><span>${label}</span><i><b style="width:${width}%"></b></i><strong>${value.toFixed(1)} ${unit}</strong></div>`;
  }

  async function loadPlacementGraphic() {
    const target = document.getElementById("placement-graphic");
    const quant = document.getElementById("placement-quant");
    const parallel = document.getElementById("placement-parallel");
    if (!target || !quant || !parallel) return;
    try {
      const [q6, q2] = await Promise.all([fetch("olmoe-q6-placement.json").then(r => r.json()), fetch("olmoe-q2-placement.json").then(r => r.json())]);
      const render = () => {
        const summary = summarizePlacement(quant.value === "q6" ? q6 : q2, Number(parallel.value));
        const maxTime = Math.max(summary.gpu.time, summary.cpu.time);
        const saved = summary.gpu.vram - summary.cpu.vram;
        const slowdown = summary.cpu.time / summary.gpu.time;
        target.innerHTML = `<div class="placement-callout"><strong>${(saved / 1024).toFixed(2)} GiB less peak VRAM</strong><span>cost ${slowdown.toFixed(1)}× median end-to-end time</span></div><div class="metric-sheet"><h3>Peak VRAM <small>placement-run scope</small></h3>${placementBar("All GPU", summary.gpu.vram, 16311, "MiB", "gpu")}${placementBar("Experts in CPU RAM", summary.cpu.vram, 16311, "MiB", "cpu")}</div><div class="metric-sheet"><h3>End-to-end time <small>${parallel.value} request${parallel.value === "1" ? "" : "s"}</small></h3>${placementBar("All GPU", summary.gpu.time, maxTime, "s", "gpu")}${placementBar("Experts in CPU RAM", summary.cpu.time, maxTime, "s", "cpu")}</div><div class="placement-detail"><span>Generation: <b>${summary.gpu.generation.toFixed(1)}</b> vs <b>${summary.cpu.generation.toFixed(1)}</b> aggregate tok/s</span><span>GPU-board energy, complete sweep: <b>${(summary.gpu.energy / 1000).toFixed(1)}</b> vs <b>${(summary.cpu.energy / 1000).toFixed(1)}</b> kJ</span></div>`;
      };
      quant.addEventListener("change", render); parallel.addEventListener("change", render); render();
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  if (document) document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-lesson]").forEach(button => button.addEventListener("click", () => selectLesson(button)));
    document.querySelectorAll("[data-evidence]").forEach(button => button.addEventListener("click", () => selectEvidence(button)));
    const firstLesson = document.querySelector("[data-lesson].active") || document.querySelector("[data-lesson]");
    if (firstLesson) selectLesson(firstLesson);
    loadBuildInfo();
    loadPlacementGraphic();
  });

  return { LESSONS, summarizePlacement };
});
