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

  if (document) document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-lesson]").forEach(button => button.addEventListener("click", () => selectLesson(button)));
    document.querySelectorAll("[data-evidence]").forEach(button => button.addEventListener("click", () => selectEvidence(button)));
    const firstLesson = document.querySelector("[data-lesson].active") || document.querySelector("[data-lesson]");
    if (firstLesson) selectLesson(firstLesson);
    loadBuildInfo();
  });

  return { LESSONS };
});
