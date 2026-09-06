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

  async function loadGemmaGraphic() {
    const target = document.getElementById("gemma-graphic");
    const parallel = document.getElementById("gemma-parallel");
    if (!target || !parallel) return;
    try {
      const data = await fetch("gemma4-q5km-offload.json").then(r => r.json());
      const render = () => {
        const row = data.requests.find(item => item.concurrency === Number(parallel.value));
        const pass = `${row.correct}/${row.requests} task answers start correctly`;
        target.innerHTML = `<div class="placement-callout"><strong>${row.aggregate_tps_mean.toFixed(2)} aggregate tok/s</strong><span>${row.per_request_tps_mean.toFixed(2)} tok/s per request</span></div><div class="metric-sheet"><h3>Throughput <small>${parallel.value} request${parallel.value === "1" ? "" : "s"}</small></h3>${placementBar("Aggregate", row.aggregate_tps_mean, 8, "tok/s", "gpu")}${placementBar("Per request", row.per_request_tps_mean, 4, "tok/s", "cpu")}</div><div class="metric-sheet"><h3>Client-observed latency <small>p50 / p95</small></h3>${placementBar("TTFT p50", row.ttft_ms_p50 / 1000, 13, "s", "gpu")}${placementBar("TTFT p95", row.ttft_ms_p95 / 1000, 13, "s", "cpu")}</div><div class="placement-detail"><span>Correctness: <b>${pass}</b></span><span>Peak: <b>${(data.telemetry.peak_vram_mib / 1024).toFixed(2)} GiB VRAM</b> · <b>${(data.telemetry.peak_process_rss_mib / 1024).toFixed(2)} GiB RSS</b></span></div>`;
      };
      parallel.addEventListener("change", render);
      render();
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  async function loadSerialGemmaGraphic() {
    const target = document.getElementById("serial-gemma-graphic");
    const batch = document.getElementById("serial-gemma-batch");
    if (!target || !batch) return;
    try {
      const data = await fetch("gemma4-q5km-serial-layer0.json").then(r => r.json());
      const render = () => {
        const row = data.runs.find(item => item.batch === Number(batch.value));
        const reuse = row.assignments / row.selected_union;
        const noReuseMb = row.stream_bytes * reuse / 1e6;
        const unionMb = row.stream_bytes / 1e6;
        target.innerHTML = `<div class="placement-callout"><strong>${row.selected_union} streams serve ${row.assignments} assignments</strong><span>${reuse.toFixed(2)} applications/fetch · ${(unionMb / row.batch).toFixed(1)} MB/request</span></div><div class="metric-sheet"><h3>Expert traffic <small>one real layer</small></h3>${placementBar("Without reuse", noReuseMb, 330, "MB", "cpu")}${placementBar("Selected union", unionMb, 330, "MB", "gpu")}</div><div class="metric-sheet"><h3>Scalar layer time <small>not end-to-end</small></h3>${placementBar("Direct Rust oracle", row.direct_ms, row.serial_ms, "ms", "gpu")}${placementBar("Ordered serial", row.serial_ms, row.serial_ms, "ms", "cpu")}</div><div class="placement-detail"><span>Serial/direct time: <b>${(row.serial_ms / row.direct_ms).toFixed(1)}×</b></span><span>Max difference: <b>${row.max_abs_error.toFixed(8)}</b> · resident parameter payload: <b>${row.resident_parameter_bytes} B</b></span></div>`;
      };
      batch.addEventListener("change", render);
      render();
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  async function loadBoundedGemmaGraphic() {
    const target = document.getElementById("bounded-gemma-graphic");
    const parallel = document.getElementById("bounded-gemma-parallel");
    if (!target || !parallel) return;
    try {
      const data = await fetch("gemma4-q5km-bounded-end-to-end.json").then(r => r.json());
      const render = () => {
        const row = data.requests.find(item => item.concurrency === Number(parallel.value));
        const telemetry = data.telemetry;
        target.innerHTML = `<div class="placement-callout"><strong>${row.correct}/${row.requests} short tasks correct</strong><span>${row.aggregate_tps_mean.toFixed(2)} aggregate · ${row.per_request_tps_mean.toFixed(2)} tok/s/request</span></div><div class="metric-sheet"><h3>Capacity <small>all-GPU allocation fails</small></h3>${placementBar("Peak VRAM", telemetry.peak_vram_mib, 16311, "MiB", "gpu")}${placementBar("GPU capacity", 16311, 16311, "MiB", "cpu")}</div><div class="metric-sheet"><h3>Reclaimable host residency <small>process RSS</small></h3>${placementBar("Minimum", telemetry.minimum_process_rss_mib, 19515, "MiB", "gpu")}${placementBar("Mean", telemetry.mean_process_rss_mib, 19515, "MiB", "gpu")}${placementBar("Peak", telemetry.peak_process_rss_mib, 19515, "MiB", "cpu")}</div><div class="placement-detail"><span>TTFT p50/p95: <b>${(row.ttft_ms_p50 / 1000).toFixed(2)} / ${(row.ttft_ms_p95 / 1000).toFixed(2)} s</b></span><span>Whole sweep: <b>${(data.expert_trace.logical_bytes / 1e9).toFixed(2)} GB logical expert bytes</b> · physical IO unknown</span></div>`;
      };
      parallel.addEventListener("change", render);
      render();
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  async function loadValidationGemmaGraphic() {
    const target = document.getElementById("validation-gemma-graphic");
    const parallel = document.getElementById("validation-gemma-parallel");
    if (!target || !parallel) return;
    try {
      const [data, logits] = await Promise.all([
        fetch("gemma4-q5km-executable-code.json").then(r => r.json()),
        fetch("gemma4-q5km-logit-equivalence.json").then(r => r.json()),
      ]);
      const render = () => {
        const concurrency = Number(parallel.value);
        const resident = data.policies.resident;
        const reclaimed = data.policies.reclaimed;
        const residentMemory = resident.telemetry.by_concurrency.find(item => item.concurrency === concurrency);
        const reclaimedMemory = reclaimed.telemetry.by_concurrency.find(item => item.concurrency === concurrency);
        const residentQuality = resident.response_quality.by_concurrency.find(item => item.concurrency === concurrency);
        const reclaimedQuality = reclaimed.response_quality.by_concurrency.find(item => item.concurrency === concurrency);
        const expert = reclaimed.expert_trace.by_concurrency.find(item => item.concurrency === concurrency);
        const strict = 100 * reclaimedQuality.strict_instruction_following / reclaimedQuality.responses;
        target.innerHTML = `<div class="placement-callout"><strong>${residentQuality.tests_passed + reclaimedQuality.tests_passed}/${residentQuality.responses + reclaimedQuality.responses} executable tests passed</strong><span>zero paired outcome disagreements · ${logits.logits.toLocaleString()} logits also bit-identical</span></div><div class="metric-sheet"><h3>Peak host RSS <small>${concurrency} request${concurrency === 1 ? "" : "s"}</small></h3>${placementBar("Resident expert pages", residentMemory.peak_process_rss_mib, 16000, "MiB", "cpu")}${placementBar("Reclaimed experts", reclaimedMemory.peak_process_rss_mib, 16000, "MiB", "gpu")}</div><div class="metric-sheet"><h3>Executable Rust <small>two repetitions</small></h3>${placementBar("Resident tests passed", 100 * residentQuality.tests_passed / residentQuality.responses, 100, "%", "gpu")}${placementBar("Reclaimed tests passed", 100 * reclaimedQuality.tests_passed / reclaimedQuality.responses, 100, "%", "gpu")}${placementBar("Strict output format", strict, 100, "%", "cpu")}</div><div class="placement-detail"><span>Reclaimed expert work: <b>${expert.assignments.toLocaleString()} assignments</b> · <b>${(expert.logical_bytes / 1e9).toFixed(2)} GB logical</b></span><span>Reclaimed peak: <b>${(reclaimedMemory.peak_process_rss_mib / 1024).toFixed(2)} GiB RSS</b> · <b>${(reclaimedMemory.peak_vram_mib / 1024).toFixed(2)} GiB VRAM</b></span></div>`;
      };
      parallel.addEventListener("change", render);
      render();
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  async function loadColdCurveGraphic() {
    const target = document.getElementById("cold-curve-graphic");
    const parallel = document.getElementById("cold-curve-parallel");
    if (!target || !parallel) return;
    try {
      const data = await Promise.all([
        fetch("gemma4-q5km-cold-hdd-c1-c8-r3.json").then(r => r.json()),
        fetch("gemma4-q5km-cold-hdd-c2-c4-r3.json").then(r => r.json()),
      ]);
      const groups = data.flatMap(item => item.groups).filter(item => item.cache_mode === "cold");
      const render = () => {
        const concurrency = Number(parallel.value);
        const resident = groups.find(item => item.concurrency === concurrency && item.policy === "resident");
        const reclaimed = groups.find(item => item.concurrency === concurrency && item.policy === "reclaimed");
        target.innerHTML = `<div class="placement-callout"><strong>${resident.tests_passed + reclaimed.tests_passed}/${resident.requests + reclaimed.requests} executable tests passed</strong><span>three cold repetitions per policy · identical Q5_K_M model</span></div><div class="metric-sheet"><h3>Passing tasks/hour <small>higher is more usable</small></h3>${placementBar("Resident pages", resident.passing_tasks_per_hour, 120, "", "cpu")}${placementBar("Reclaimed pages", reclaimed.passing_tasks_per_hour, 120, "", "gpu")}</div><div class="metric-sheet"><h3>Peak host RSS <small>lower is better</small></h3>${placementBar("Resident pages", resident.peak_rss_mib_mean / 1024, 16, " GiB", "cpu")}${placementBar("Reclaimed pages", reclaimed.peak_rss_mib_mean / 1024, 16, " GiB", "gpu")}</div><div class="placement-detail"><span>Physical reads: <b>${(resident.request_read_bytes_mean / 1e9).toFixed(2)} GB</b> · <b>${(resident.request_read_bytes_per_passing_task / 1e9).toFixed(2)} GB/passing task</b></span><span>Aggregate generation: <b>${resident.generated_tokens_per_second.toFixed(2)} tok/s resident</b> · <b>${reclaimed.generated_tokens_per_second.toFixed(2)} tok/s reclaimed</b></span></div>`;
      };
      parallel.addEventListener("change", render);
      render();
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  async function loadStreamTierGraphic() {
    const target = document.getElementById("stream-tier-graphic");
    if (!target) return;
    try {
      const data = await fetch("gemma4-stream-tier-analysis.json").then(r => r.json());
      const find = (tier, backend) => data.groups.find(row => row.tier === tier && row.cache === "cold" && row.backend === backend);
      const hdd = find("hdd", "prefetch");
      const nvme = find("nvme", "prefetch");
      const sync = find("hdd", "sync");
      target.innerHTML = `<div class="placement-callout"><strong>${(hdd.bytes / 1e6).toFixed(2)} MB through a 2 MiB buffer</strong><span>identical digest in ${data.groups.reduce((sum, row) => sum + row.runs, 0)} replays</span></div><div class="metric-sheet"><h3>Cold payload rate <small>higher is better</small></h3>${placementBar("HDD sequential", hdd.bandwidth_mib_s.p50, 900, "MiB/s", "cpu")}${placementBar("NVMe control", nvme.bandwidth_mib_s.p50, 900, "MiB/s", "gpu")}</div><div class="metric-sheet"><h3>HDD physical request shape <small>large forward reads</small></h3>${placementBar("Synchronous", sync.device_average_read_bytes.p50 / 1e6, 4, "MB/read", "cpu")}${placementBar("Double buffer", hdd.device_average_read_bytes.p50 / 1e6, 4, "MB/read", "gpu")}</div><div class="placement-detail"><span>Cold HDD: <b>${hdd.elapsed_ms.p50.toFixed(1)} ms</b> · <b>${hdd.device_read_ios.p50} device reads</b></span><span>Prefetch gain: HDD <b>${(100 * (sync.elapsed_ms.p50 - hdd.elapsed_ms.p50) / sync.elapsed_ms.p50).toFixed(1)}%</b> · storage remains the bottleneck</span></div>`;
    } catch (_) { target.innerHTML = "<p>Measured JSON unavailable in this preview.</p>"; }
  }

  if (document) document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-lesson]").forEach(button => button.addEventListener("click", () => selectLesson(button)));
    document.querySelectorAll("[data-evidence]").forEach(button => button.addEventListener("click", () => selectEvidence(button)));
    const firstLesson = document.querySelector("[data-lesson].active") || document.querySelector("[data-lesson]");
    if (firstLesson) selectLesson(firstLesson);
    loadBuildInfo();
    loadPlacementGraphic();
    loadGemmaGraphic();
    loadSerialGemmaGraphic();
    loadBoundedGemmaGraphic();
    loadValidationGemmaGraphic();
    loadColdCurveGraphic();
    loadStreamTierGraphic();
  });

  return { LESSONS, summarizePlacement };
});
